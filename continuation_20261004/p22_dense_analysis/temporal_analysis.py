"""Read-only, receipt-verified P22 field analysis. No images, labels, torch or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path('/home/chenhc/claude_try_MDMOT/continuation_20261004/p22_dense_motion')
OUT = Path(__file__).resolve().parent
PAIRS = ['23', '25', '29', '69', '78']
REGIONS = ['all', 'edge', 'interior']
BINS = ['lt16', '16to32', '32to64', 'ge64']
EPS = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''):
            h.update(x)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify_receipt():
    receipt_path = ROOT / 'FEATURE_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    paths = []
    for row in receipt['files']:
        path = Path(row['path'])
        assert path.parent == ROOT and path.is_file(), path
        assert sha(path) == row['sha256'], path
        paths.append(path)
    assert len(paths) == len(set(paths)) == 45
    assert sha(ROOT / 'PROTOCOL.json') == receipt['protocol_sha256']
    assert receipt['PID_read'] is False and receipt['model_trained'] is False
    assert receipt['cal_dev_val_test_read'] is False
    npzs = sorted(p for p in paths if p.suffix == '.npz')
    assert len(npzs) == 20 and all(p.with_suffix('.json') in paths for p in npzs)
    return receipt, npzs, {'receipt_path': str(receipt_path), 'receipt_sha256': sha(receipt_path),
                         'verified_files': len(paths), 'all_hashes_match': True}


def project(matrix, points):
    homogeneous = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1) @ matrix.T
    denominator = homogeneous[..., 2:3]
    valid = np.isfinite(homogeneous).all(-1) & (np.abs(denominator[..., 0]) > EPS)
    value = homogeneous[..., :2] / np.where(np.abs(denominator) > EPS, denominator, 1.)
    return value, valid & np.isfinite(value).all(-1)


def roi_points(box):
    # Match the archived extractor's float32 ROI-grid generation.
    x1, y1, x2, y2 = box
    x = x1 + (np.arange(16, dtype=np.float32) + .5) / 16 * max(x2-x1, 0)
    y = y1 + (np.arange(16, dtype=np.float32) + .5) / 16 * max(y2-y1, 0)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1).astype(np.float64)


def pull_previous(previous, current, points, hcurrent, hprevious):
    background_current, good1 = project(hcurrent, points)
    # These are each grid point's traced previous positions, not box centers.
    previous_points = background_current + current
    background_previous, good2 = project(hprevious, previous_points)
    past_endpoint = background_previous + previous
    inverse = np.linalg.inv(hprevious)
    pulled_actual, good3 = project(inverse, past_endpoint)
    pulled_background, good4 = project(inverse, background_previous)
    residual = pulled_actual - pulled_background
    good = good1 & good2 & good3 & good4
    good &= np.isfinite(previous_points).all(-1) & np.isfinite(past_endpoint).all(-1)
    good &= np.isfinite(previous).all(-1) & np.isfinite(current).all(-1) & np.isfinite(residual).all(-1)
    roundtrip_error = np.linalg.norm(pulled_background - previous_points, axis=-1)
    return residual, previous_points, good, roundtrip_error


def design(points, box, degree):
    width, height = max(box[2]-box[0], EPS), max(box[3]-box[1], EPS)
    x = (points[..., 0] - (box[0]+box[2])/2) / width * 2
    y = (points[..., 1] - (box[1]+box[3])/2) / height * 2
    terms = [np.ones_like(x)]
    if degree >= 1:
        terms.extend([x, y])
    if degree >= 2:
        terms.extend([x*x, x*y, y*y])
    return np.stack(terms, axis=-1)


def fit_predict(train_x, train_y, test_x):
    if len(train_x) < train_x.shape[1] or np.linalg.matrix_rank(train_x) < train_x.shape[1]:
        return None
    coefficients = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
    return test_x @ coefficients


def mse(a, b):
    return float(np.mean(np.sum((a-b)**2, axis=-1)))


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a*b)/den) if den > EPS else None


def spatial_metrics(current, x1, x2):
    if len(current) < 3:
        return {}
    centered = current-current.mean(0)
    affine = fit_predict(x1, current, x1)
    quadratic = fit_predict(x2, current, x2)
    var = mse(current, current.mean(0))
    s = np.linalg.svd(centered, compute_uv=False)
    energy = float(np.sum(s*s))
    result = {'current_mean_error_mse': var, 'current_shape_rms_pixels': float(np.sqrt(var)),
              'current_mean_vector_magnitude_pixels': float(np.linalg.norm(current.mean(0))),
              'directional_rank1_energy_fraction': float(s[0]**2/energy) if energy > EPS else None}
    if affine is not None:
        err = mse(current, affine)
        result.update(current_affine_error_mse=err, current_affine_residual_rms_pixels=float(np.sqrt(err)),
                      affine_fraction_of_centered_energy=1-err/var if var > EPS else None)
    if quadratic is not None:
        err = mse(current, quadratic)
        result.update(current_quadratic_error_mse=err,
                      quadratic_fraction_of_centered_energy=1-err/var if var > EPS else None)
    return result


def temporal_metrics(current, previous, xc1, xp1, xp2):
    if len(current) < 3:
        return {}
    cm, pm = current.mean(0), previous.mean(0)
    cc, pc = current-cm, previous-pm
    full, constant = mse(current, previous), mse(current, pm)
    centered_full, centered_zero = mse(cc, pc), mse(cc, np.zeros_like(cc))
    result = {'prior_full_error_mse': full, 'prior_constant_error_mse': constant,
              'prior_full_minus_constant_mse': full-constant,
              'prior_full_better_than_constant': float(full < constant),
              'centered_shape_cosine': cosine(cc, pc),
              'centered_prior_error_mse': centered_full,
              'centered_zero_error_mse': centered_zero,
              'centered_prior_minus_zero_mse': centered_full-centered_zero,
              'centered_prior_better_than_zero': float(centered_full < centered_zero),
              'mean_vector_temporal_delta_pixels': float(np.linalg.norm(cm-pm)),
              'current_centered_rms_pixels': float(np.sqrt(centered_zero)),
              'previous_centered_rms_pixels': float(np.sqrt(mse(pc, np.zeros_like(pc))))}
    prior_affine = fit_predict(xp1, previous, xp1)
    current_affine = fit_predict(xc1, current, xc1)
    prior_quadratic = fit_predict(xp2, previous, xp2)
    if prior_affine is not None:
        aff = mse(current, prior_affine)
        result.update(prior_affine_error_mse=aff, prior_full_minus_affine_mse=full-aff,
                      prior_full_better_than_affine=float(full < aff))
    if prior_quadratic is not None:
        quad = mse(current, prior_quadratic)
        result.update(prior_quadratic_error_mse=quad, prior_full_minus_quadratic_mse=full-quad,
                      prior_full_better_than_quadratic=float(full < quad))
    if prior_affine is not None and current_affine is not None:
        cd, pd = current-current_affine, previous-prior_affine
        zero, pred = mse(cd, np.zeros_like(cd)), mse(cd, pd)
        result.update(affine_detrended_cosine=cosine(cd, pd),
                      affine_detrended_current_rms_pixels=float(np.sqrt(zero)),
                      affine_detrended_prior_minus_zero_mse=pred-zero,
                      affine_detrended_prior_better_than_zero=float(pred < zero))
    return result


def block_holdout(current, previous, mask, xc1, xc2, xp1, xp2):
    # Four contiguous 8x8 ROI-index quadrants. No test current values enter fitting.
    yy, xx = np.indices((16, 16))
    blocks = 2*(yy >= 8)+(xx >= 8)
    if mask.sum() < 6:
        return {}
    prior_affine = fit_predict(xp1[mask], previous[mask], xp1.reshape(-1, 3))
    prior_quadratic = fit_predict(xp2[mask], previous[mask], xp2.reshape(-1, 6))
    if prior_affine is None or prior_quadratic is None:
        return {}
    prior_affine = prior_affine.reshape(16, 16, 2)
    prior_quadratic = prior_quadratic.reshape(16, 16, 2)
    names = ['current_mean', 'current_affine', 'current_quadratic',
             'prior_full_offset', 'prior_affine_offset', 'prior_quadratic_offset']
    totals = {k: 0. for k in names}
    count, folds = 0, 0
    for block in range(4):
        train, test = mask & (blocks != block), mask & (blocks == block)
        if not test.any():
            continue
        ca = fit_predict(xc1[train], current[train], xc1[test])
        cq = fit_predict(xc2[train], current[train], xc2[test])
        if ca is None or cq is None:
            continue
        mean = current[train].mean(0)
        predictions = {'current_mean': mean, 'current_affine': ca, 'current_quadratic': cq}
        for name, field in [('prior_full_offset', previous), ('prior_affine_offset', prior_affine),
                            ('prior_quadratic_offset', prior_quadratic)]:
            predictions[name] = field[test] + mean-field[train].mean(0)
        for name, pred in predictions.items():
            totals[name] += float(np.sum((current[test]-pred)**2))
        count += int(test.sum())
        folds += 1
    if not count:
        return {}
    out = {k+'_mse': v/count for k, v in totals.items()}
    out.update(test_grid_samples=count, valid_folds=folds)
    for baseline in ['current_mean', 'current_affine', 'current_quadratic', 'prior_affine_offset', 'prior_quadratic_offset']:
        delta = out['prior_full_offset_mse']-out[baseline+'_mse']
        out['prior_full_minus_'+baseline+'_mse'] = delta
        out['prior_full_better_than_'+baseline] = float(delta < 0)
    return out


def stats(values):
    x = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(x):
        return {'n': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None}
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p10': float(np.percentile(x, 10)), 'p90': float(np.percentile(x, 90))}


def aggregate(rows):
    result = {'objects': len(rows), 'support': {k: stats([r['support'][k] for r in rows])
              for k in rows[0]['support']} if rows else {}, 'spatial': {}, 'temporal': {}}
    for category in ['spatial', 'temporal']:
        for region in REGIONS:
            keys = sorted({k for r in rows for k in r[category][region]})
            result[category][region] = {k: stats([r[category][region].get(k) for r in rows]) for k in keys}
    keys = sorted({k for r in rows for k in r['block_holdout']})
    result['block_holdout'] = {k: stats([r['block_holdout'].get(k) for r in rows]) for k in keys}
    result['corrected_quantity_gate_objects'] = sum(r['corrected_quantity_gate_object'] for r in rows)
    return result


def concentration(rows, baseline):
    usable = [r for r in rows if 'prior_full_offset_mse' in r['block_holdout']]
    gain = np.array([r['block_holdout'][baseline+'_mse']-r['block_holdout']['prior_full_offset_mse']
                     for r in usable], dtype=float)
    positive = np.maximum(gain, 0)
    n = int(np.ceil(len(positive)/10))
    return {'objects': len(usable), 'gain_mse': stats(gain.tolist()),
            'objects_with_positive_gain': int((gain > 0).sum()),
            'top_decile_objects': n,
            'top_decile_share_of_positive_gain': float(np.sort(positive)[-n:].sum()/positive.sum()) if positive.sum() > 0 else None,
            'scope': 'posthoc descriptive concentration, no selection or tuned threshold'}


def operator_checks():
    rng = np.random.default_rng(20261004)
    points = rng.uniform(100, 500, (31, 2))
    hc = np.array([[1.02, .05, 10.], [-.03, .98, -6.], [.0002, -.0001, 1.]])
    hp = np.array([[.95, -.07, -8.], [.02, 1.04, 3.], [-.0001, .0003, 1.]])
    rc = rng.normal(0, 2, points.shape)
    pp = project(hc, points)[0]+rc
    wanted = rng.normal(0, .5, points.shape)
    rp = project(hp, pp+wanted)[0]-project(hp, pp)[0]
    got, traced, valid, error = pull_previous(rp, rc, points, hc, hp)
    assert valid.all() and np.max(np.abs(got-wanted)) < 1e-9
    assert np.max(np.abs(traced-pp)) < 1e-9 and error.max() < 1e-9
    train = np.column_stack([np.ones(8), np.arange(8), np.arange(8)**2])
    target = train @ rng.normal(size=(3, 2))
    assert np.max(np.abs(fit_predict(train, target, train)-target)) < 1e-9
    return {'exact_projective_endpoint_pullback': True, 'actual_traced_points_used': True,
            'affine_quadratic_least_squares': True, 'GPU_used': False}


def number(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def render(result):
    lines = ['# P22 稠密残差场空间与两步时间分析', '',
             '**状态：输入观测诊断；不能把原始数量门 gate_pass=true 写成方法 PASS。**', '',
             '**信号判断：保留受限输入信号。** 原数量条件下共同支持的267个对象区域，其非仿射空间形状在邻步高度复现，并且在固定空间留出中优于常量与低维场；这不是单纯std大。但其余对象多数场幅值很小，总体改善由高幅值尾部与bbox边缘主导。尚未证实这些形状来自目标自身动力学，也没有跨视图或身份信息证据，表示学习保持HOLD。', '',
             '只读经收据校验的 20 个 NPZ 与对应 JSON；45 项输入哈希全部匹配。没有读取图像、XML、身份标签，未使用 GPU、重提取或训练。', '',
             '前一残差的终点在 t-2 坐标系，本分析用每个当前网格点的 `p_prev=H_current(p)+r_current` 重建轨迹位置，再用 `H_previous^-1(H_previous(p_prev)+r_previous)-p_prev` 精确拉回 t-1；使用完整投影映射，没有使用单个框中心 Jacobian。', '',
             '时间支持严格是 `valid[0] & valid[1]` 加全部有限端点/投影条件。旧 `visible_fraction` 是两步均值；它不等于共同有效比例。', '',
             '## 按 pair 的空间信息与时间形状', '',
             '每行先对每个对象计算，再给出对象中位数；rmse 单位为 native pixel。affine 残差是相对最佳六参数二维仿射向量场。shape cosine 去掉每个对象的常量均值；non-affine cosine 进一步去掉各自最佳仿射场。', '',
             '| pair | 对象 | joint ≥.5 | 修正数量门 | shape RMS | affine后RMS | centered cosine | non-affine cosine | prior full优于constant | prior full优于affine |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for pair in PAIRS:
        a = result['per_pair'][pair]
        s, t = a['spatial']['all'], a['temporal']['all']
        rows = [r for r in result['objects'] if r['pair'] == pair]
        vals = [pair, len(rows), sum(r['support']['joint_valid_fraction'] >= .5 for r in rows),
                a['corrected_quantity_gate_objects'], number(s['current_shape_rms_pixels']['median']),
                number(s['current_affine_residual_rms_pixels']['median']),
                number(t['centered_shape_cosine']['median']), number(t['affine_detrended_cosine']['median']),
                number(t['prior_full_better_than_constant']['mean']), number(t['prior_full_better_than_affine']['mean'])]
        lines.append('| '+' | '.join(map(str, vals))+' |')
    lines += ['', '## 原数量条件下的真实时间与非仿射信号', '',
              '仅复用原有 `joint>=.5` 且 `joint当前std>原背景p90` 条件作描述分层，没有新增或调节阈值。该条件消费当前RAFT输出，因此不是独立选择或确认性评估。', '',
              '| pair | 条件内对象 | shape RMS中位 | affine后RMS中位 | centered cosine中位 | non-affine cosine中位 | holdout full胜current affine | holdout full胜prior affine |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for pair in PAIRS:
        a = result['existing_quantity_condition']['per_pair'][pair]['condition_met']
        s, t, c = a['spatial']['all'], a['temporal']['all'], a['block_holdout']
        lines.append('| '+' | '.join([pair, str(a['objects']), number(s['current_shape_rms_pixels']['median']),
                      number(s['current_affine_residual_rms_pixels']['median']), number(t['centered_shape_cosine']['median']),
                      number(t['affine_detrended_cosine']['median']),
                      number(c['prior_full_better_than_current_affine']['mean']),
                      number(c['prior_full_better_than_prior_affine_offset']['mean'])])+' |')
    met = result['existing_quantity_condition']['all']['condition_met']
    rest = result['existing_quantity_condition']['all']['condition_not_met']
    lines += ['', f"合计条件内 {met['objects']} 个区域：去均值cosine中位 {number(met['temporal']['all']['centered_shape_cosine']['median'])}、去仿射后 {number(met['temporal']['all']['affine_detrended_cosine']['median'])}；其余 {rest['objects']} 个区域的空间RMS中位仅 {number(rest['spatial']['all']['current_shape_rms_pixels']['median'])} px。条件内不是按身份或预测错误挑出的集合；也不是所有对象的代表。"]
    lines += ['', '## 四象限空间留出', '',
              '固定四个连续 8×8 网格象限，轮流留出一个象限。current mean/affine/quadratic 仅用其余三个象限的当前场拟合。prior field/affine/quadratic 可使用所有既往场，再仅从训练象限估计一个常量 offset；测试当前值不参与拟合。这回答旧场的空间形状是否帮助重建留出的当前场。', '',
              '小框中的象限可小于 RAFT 的 8 像素特征跨度；留出不使共享卷积特征独立。16×16 采样位置也不是256个独立运动观测。', '',
              '表中为先计算对象 MSE、再在pair内取均值；使用相同成功fold支持。', '',
              '| pair | current mean MSE | current affine | current quadratic | prior full+offset | prior affine+offset | prior quadratic+offset | full优于current affine比例 | full优于prior affine比例 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for pair in PAIRS:
        c = result['per_pair'][pair]['block_holdout']
        names = ['current_mean_mse', 'current_affine_mse', 'current_quadratic_mse',
                 'prior_full_offset_mse', 'prior_affine_offset_mse', 'prior_quadratic_offset_mse',
                 'prior_full_better_than_current_affine', 'prior_full_better_than_prior_affine_offset']
        lines.append('| '+pair+' | '+' | '.join(number(c[k]['mean']) for k in names)+' |')
    gain = result['gain_concentration']['all']['prior_affine_offset']
    lines += ['', f"总体相对 prior affine+offset 的正向MSE改善中，最高10%对象贡献 {number(100*gain['top_decile_share_of_positive_gain'],2)}%；只有 {gain['objects_with_positive_gain']}/{gain['objects']} 对象改善。均值改善与典型对象改善必须分开表述。"]
    lines += ['', '## Native短边分层', '',
              '| 原始短边 | 对象 | 中位joint支持 | 中位stride8覆盖格 | shape RMS | affine后RMS | centered cosine | non-affine cosine |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in BINS:
        a = result['per_native_bin'][b]
        if not a['objects']:
            continue
        s, t = a['spatial']['all'], a['temporal']['all']
        lines.append('| '+' | '.join([b, str(a['objects']), number(a['support']['joint_valid_fraction']['median']),
                      number(a['support']['current_visible_stride8_cells']['median'], 1),
                      number(s['current_shape_rms_pixels']['median']), number(s['current_affine_residual_rms_pixels']['median']),
                      number(t['centered_shape_cosine']['median']), number(t['affine_detrended_cosine']['median'])])+' |')
    lines += ['', '## 边缘与内部', '',
              'interior 是 ROI 中间 8×8 点，其余192点是 bbox edge，非语义分割前景/背景。仅在对应区域共同可见点上计算，完整分pair/分尺度数据见JSON。', '',
              '| 区域 | current shape RMS中位 | affine后RMS中位 | centered cosine中位 | non-affine cosine中位 | prior full-constant MSE均值 |',
              '|---|---:|---:|---:|---:|---:|']
    for region in REGIONS:
        s, t = result['all_objects']['spatial'][region], result['all_objects']['temporal'][region]
        lines.append('| '+' | '.join([region, number(s['current_shape_rms_pixels']['median']),
                      number(s['current_affine_residual_rms_pixels']['median']), number(t['centered_shape_cosine']['median']),
                      number(t['affine_detrended_cosine']['median']), number(t['prior_full_minus_constant_mse']['mean'])])+' |')
    lines += ['', '## 解释边界', '',
              '- 当前 RAFT flow 决定回溯采样位置、共同可见mask和当前 residual；前后邻帧也共享一帧图像与同一冻结RAFT。因此本结果是**共享RAFT轨迹条件下的场复现诊断**，不是独立因果预测或GT光流误差。常量offset版本还消费部分当前场，只是空间留出重建。',
              '- 封存NPZ没有实际 prior_points、全分辨率flow、图像尺寸或前后向误差数组。按保存H与residual重建轨迹；原始 OpenCV raster插值与float32舍入造成的微小端点差异不能从该cache完全重建。精确投影拉回不等于拥有独立或无噪声轨迹真值。',
              '- 最佳affine/quadratic的全支持拟合是低维解释率上界；directional rank1仅表示二维运动方向协方差，不是空间独立样本数。stride8格数只是几何覆盖数量。',
              '- 当前场std超过背景p90只是数量/幅度筛查，不验证对象前景、非线性空间信息、时间可迁移或身份互补。修正共同支持后仍达数量门也不能把输入门升级成方法PASS。',
              '- 五个pair只有两个anchor和两个view；网格、对象与相邻帧高度相关。本报告不计算伪独立像素显著性，不提供MDA/IDF1/MOTA，也不据该诊断改变阈值。', '',
              '具体信号判断写在 `signal_decision` 字段；该字段仅总结本次诊断的观测支持，不是训练或官方评测授权。', '']
    return '\n'.join(lines)


def main():
    receipt, paths, verification = verify_receipt()
    tests = operator_checks()
    rows, tasks = [], []
    yy, xx = np.indices((16,16))
    interior = (xx >= 4) & (xx < 12) & (yy >= 4) & (yy < 12)
    regions = {'all': np.ones((16,16), bool), 'interior': interior, 'edge': ~interior}
    for path in paths:
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['pair'] in PAIRS and meta['PID_read'] is False and meta['raw_XML_read'] is False
        anchor = meta['anchor']
        hc = np.array(meta['background_matrices'][str(anchor)], dtype=float)
        hp = np.array(meta['background_matrices'][str(anchor-1)], dtype=float)
        assert hc.shape == hp.shape == (3,3)
        with np.load(path, allow_pickle=False) as data:
            fields = data['residual_fields'].astype(np.float64)
            masks = data['valid'].astype(bool)
            keys = data['keys'].tolist()
        assert fields.shape == (len(keys),2,16,16,2) and masks.shape == (len(keys),2,16,16)
        assert keys == [o['key'] for o in meta['objects']]
        tasks.append({'npz': str(path), 'pair': meta['pair'], 'camera': meta['camera'], 'anchor': anchor,
                      'objects': len(keys), 'background': meta['geometry']})
        for obj, field, mask in zip(meta['objects'], fields, masks):
            box = obj['bbox']; p = roi_points(box)
            previous, current = field
            pulled, pp, finite, roundtrip = pull_previous(previous, current, p, hc, hp)
            current_valid = mask[1] & np.isfinite(current).all(-1) & np.isfinite(p).all(-1)
            joint = mask[0] & current_valid & finite
            xc1, xc2 = design(p, box, 1), design(p, box, 2)
            xp1, xp2 = design(pp, box, 1), design(pp, box, 2)
            side = obj['native_min_side']
            b = BINS[0] if side < 16 else BINS[1] if side < 32 else BINS[2] if side < 64 else BINS[3]
            spatial, temporal = {}, {}
            for region, rmask in regions.items():
                c, j = current_valid & rmask, joint & rmask
                spatial[region] = spatial_metrics(current[c], xc1[c], xc2[c])
                temporal[region] = temporal_metrics(current[j], pulled[j], xc1[j], xp1[j], xp2[j])
            noise = obj['background_holdout_p90_pixels']
            jrms = float(np.sqrt(mse(current[joint], current[joint].mean(0)))) if joint.any() else None
            support = {'native_min_side': side, 'current_valid_fraction': float(current_valid.mean()),
                       'joint_valid_fraction': float(joint.mean()), 'stored_average_valid_fraction': float(mask.mean()),
                       'current_visible_native_pixel_cells': int(len(np.unique(np.floor(p[current_valid]).astype(int), axis=0))),
                       'current_visible_stride8_cells': int(len(np.unique(np.floor(p[current_valid]/8).astype(int), axis=0))),
                       'joint_visible_stride8_cells': int(len(np.unique(np.floor(p[joint]/8).astype(int), axis=0))),
                       'quadrant_short_side_native_pixels': side/2,
                       'inverse_roundtrip_max_pixels': float(roundtrip[finite].max()) if finite.any() else None,
                       'exact_pullback_minus_unaligned_previous_rms': float(np.sqrt(mse(pulled[joint], previous[joint]))) if joint.any() else None,
                       'background_holdout_p90_pixels': noise,
                       'joint_current_shape_rms_pixels': jrms}
            rows.append({'key': obj['key'], 'pair': meta['pair'], 'camera': meta['camera'], 'anchor': anchor,
                         'native_bin': b, 'support': support, 'spatial': spatial, 'temporal': temporal,
                         'block_holdout': block_holdout(current, pulled, joint, xc1, xc2, xp1, xp2),
                         'corrected_quantity_gate_object': bool(joint.mean() >= .5 and jrms is not None and noise is not None and jrms > noise)})
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'P22_FIELD_DIAGNOSTIC_COMPLETE_NOT_METHOD_PASS', 'verification': verification,
              'analysis_code_sha256': sha(Path(__file__)), 'operator_checks': tests,
              'scope': {'raw_images_read': False, 'XML_PID_read': False, 'GPU_used': False,
                        'training': False, 'reextraction': False, 'threshold_tuning': False},
              'definitions': {'variance_unit': 'mean squared native-pixel vector error, per object',
                              'summary_unit': 'objects nested in pair/anchor/view; pair-level summaries; no pixel iid inference',
                              'temporal_alignment': 'pp=H_current(p)+r_current; H_previous^-1(H_previous(pp)+r_previous)-pp',
                              'temporal_support': 'valid[0] AND valid[1] AND all finite projection and residual endpoints',
                              'low_rank_controls': 'constant 2 parameters; affine 6; quadratic 12; no tuning',
                              'spatial_holdout': 'four fixed contiguous 8x8 ROI-index quadrants',
                              'prior_prediction_scope': 'shared-RAFT trajectory-conditioned diagnostic, not causal independent prediction'},
              'tasks': tasks, 'objects': rows, 'all_objects': aggregate(rows),
              'per_pair': {p: aggregate([r for r in rows if r['pair']==p]) for p in PAIRS},
              'per_native_bin': {b: aggregate([r for r in rows if r['native_bin']==b]) for b in BINS},
              'per_pair_native_bin': {p: {b: aggregate([r for r in rows if r['pair']==p and r['native_bin']==b]) for b in BINS} for p in PAIRS}}
    result['existing_quantity_condition'] = {
        'definition': 'unchanged .5 joint support and current joint std > saved background p90; descriptive only',
        'all': {name: aggregate([r for r in rows if r['corrected_quantity_gate_object'] == value])
                for name, value in [('condition_met', True), ('condition_not_met', False)]},
        'per_pair': {p: {name: aggregate([r for r in rows if r['pair']==p and r['corrected_quantity_gate_object'] == value])
                         for name, value in [('condition_met', True), ('condition_not_met', False)]} for p in PAIRS}}
    baselines = ['current_mean', 'current_affine', 'prior_affine_offset', 'prior_quadratic_offset']
    result['gain_concentration'] = {'all': {b: concentration(rows,b) for b in baselines},
        'per_pair': {p: {b: concentration([r for r in rows if r['pair']==p],b) for b in baselines} for p in PAIRS}}
    met = result['existing_quantity_condition']['all']['condition_met']
    result['signal_decision'] = {
        'status': 'KEEP_CONDITIONAL_NONAFFINE_TEMPORAL_INPUT_SIGNAL_HOLD_REPRESENTATION',
        'method_pass': False, 'quantity_gate_alone_is_sufficient': False,
        'positive_evidence': [
            'On the original quantity-conditioned 267 regions, centered and affine-detrended shapes reproduce across adjacent steps.',
            'The prior full field with a current-train-block constant offset beats affine and quadratic summaries in pair-mean held-out current-field error on all five pairs.',
            'Thus the informative subset cannot be dismissed as a repeated scalar vector or entirely affine field.'
        ],
        'condition_met_count': met['objects'],
        'condition_met_centered_cosine_median': met['temporal']['all']['centered_shape_cosine']['median'],
        'condition_met_nonaffine_cosine_median': met['temporal']['all']['affine_detrended_cosine']['median'],
        'condition_met_holdout_full_beats_current_affine_fraction': met['block_holdout']['prior_full_better_than_current_affine']['mean'],
        'condition_met_holdout_full_beats_prior_affine_fraction': met['block_holdout']['prior_full_better_than_prior_affine_offset']['mean'],
        'limits': [
            'Most unconditioned regions have tiny field amplitude; the mean benefit is highly tail-concentrated and stronger at bbox edges than interiors.',
            'Observed structure may describe stable foreground-background mixing or motion boundaries, not pure object-internal dynamics.',
            'Current RAFT trajectories and visibility condition the diagnostic; prior and current estimates share an image and model.',
            'No raw RGB-independent truth, long-term predictor, cross-view transfer, identity information or tracking gain was established.',
            'No trained representation, novelty claim, automatic training authorization or method PASS follows.'
        ]}
    OUT.mkdir(exist_ok=True)
    dump(OUT/'TEMPORAL_ANALYSIS.json', result)
    (OUT/'TEMPORAL_ANALYSIS.md').write_text(render(result))
    dump(OUT/'TEMPORAL_ANALYSIS_RECEIPT.json', {'status': result['status'], 'input_verification': verification,
         'files': [{'path': str(p), 'sha256': sha(p)} for p in [Path(__file__),OUT/'TEMPORAL_ANALYSIS.json',OUT/'TEMPORAL_ANALYSIS.md']]})
    print(json.dumps({'status': result['status'], 'objects': len(rows),
                      'corrected_counts': {p: result['per_pair'][p]['corrected_quantity_gate_objects'] for p in PAIRS},
                      'operator_checks': tests, 'verification': verification}))


if __name__ == '__main__':
    main()
