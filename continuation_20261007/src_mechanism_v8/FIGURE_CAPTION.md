SRC mechanism. (a) Illustrative five-class historical track and incoming detection evidence; the implementation aggregates semantic evidence into human and vehicle groups. (b) Analytic disagreement responses under the saved prior π(H)=0.332. Dark curves represent certain H/V beliefs; faint curves interpolate those beliefs halfway toward the prior. Prior-like belief yields zero penalty. Hollow circles locate the two conflicting real-case pairs from (c). The penalty follows the equation defined in the method section; reliability affects only the post-commitment belief update. (c) The same closed calibration candidate component before and after cost refinement. Both matrices share the [0,1] color scale. Solid and dashed boxes mark selected same-owner and different-owner pairs; crosses mark costs above the unchanged threshold of 0.80. Two tied base optima become one unique compatible optimum.

LaTeX insertion template (the standalone `.tex` copy remains local under the existing repository policy):

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{src_assignment_mechanism_v8/src_semantic_response_triptych.pdf}
\caption{SRC mechanism. (a) Illustrative five-class track and detection evidence; the implementation aggregates evidence into human and vehicle groups. (b) Analytic responses of Eq.~\eqref{eq:penalty} with $\pi(H)=0.332$. Dark curves represent certain H/V beliefs; faint curves move those beliefs halfway toward the prior. Prior-like belief yields zero penalty. Circles locate the conflicting real-case pairs from (c). (c) The same closed calibration candidate component before and after cost refinement, on a shared $[0,1]$ scale. Solid and dashed boxes mark selected same-owner and different-owner pairs; crosses mark costs above the unchanged threshold of $0.80$. Two tied base optima become one unique compatible optimum.}
\label{fig:src_semantic_cost_mechanism}
\end{figure*}
```
