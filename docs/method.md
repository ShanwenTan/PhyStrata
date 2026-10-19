# Method

PhyStrata changes the aggregation of an existing drift loss. It leaves the denoiser's noise schedule and prediction target, the temporal predictor objective, and inference guidance intact. The implementation is in `finetrainers/models/wan/physical_motion_phystrata.py`.

## From real video to activity

Let a frozen video VAE produce clean latent video $Z_0\in\mathbb{R}^{B\times T\times C\times H\times W}$. With temporal offset $\tau=2$ and $K=T-\tau$, define

$$
D_{b,i}=Z_{0,b,i+\tau}-Z_{0,b,i},\qquad
a_{b,i}=\sqrt{\operatorname{mean}_{C,H,W} D_{b,i}^{2}},\qquad
A_b=\frac{1}{K}\sum_{i=1}^{K}a_{b,i}.
$$

$a_{b,i}$ is a nonnegative latent activity magnitude. It does not identify velocity, direction, or physical correctness. All weight calculations use detached float32 real-video latents.

## Video-level weighting

At each training step, distributed `all_gather` collects the $A_b$ values from every rank. The resulting global batch of $N$ videos is the reference pool. For each local video, ties receive a midrank:

$$
p_b^{\mathrm{vid}}=\frac{\#\{A<A_b\}+\tfrac12\#\{A=A_b\}}{N},\qquad
W_b=1+\beta_v(2p_b^{\mathrm{vid}}-1),\quad \beta_v=0.4.
$$

This is the **global-batch reference** used by the released variant, not a fixed dataset-wide percentile. Its comparison pool is 32 for both the reference 2B and 5B recipes. $W_b$ stays within $[0.6,1.4]$.

## Symmetric frame credit

Each latent position receives the mean activity of the offset-$\tau$ intervals touching it. Boundary positions use the available interval:

$$
\mathcal N(j)=\{i:j=i\text{ or }j=i+\tau\},\qquad
c_{b,j}=\frac{1}{|\mathcal N(j)|}\sum_{i\in\mathcal N(j)} a_{b,i}.
$$

Within a video, midrank the $T$ credit values and emphasize both tails:

$$
p_{b,j}^{\mathrm{frm}}=\frac{\#\{c<c_{b,j}\}+\tfrac12\#\{c=c_{b,j}\}}{T},\qquad
h_{b,j}=(2p_{b,j}^{\mathrm{frm}}-1)^2.
$$

A reliability gate suppresses rank differences when frame activity is nearly flat:

$$
d_b=\frac{\sqrt{T^{-1}\sum_j(c_{b,j}-\bar c_b)^2}}{\max(\bar c_b,10^{-8})},\qquad
\eta_b=\frac{d_b}{d_b+\kappa},\quad\kappa=0.07.
$$

Both endpoints contribute equally to interval credit, which is centered to keep the within-video average unchanged:

$$
g_{b,i}=\tfrac12(h_{b,i}+h_{b,i+\tau}),\qquad
U_{b,i}=1+\beta_f\eta_b(g_{b,i}-\bar g_b),\quad\beta_f=0.4.
$$

For each video, $K^{-1}\sum_iU_{b,i}=1$, with $U_{b,i}\in[0.6,1.4]$. When $T<2\tau$, frame weighting returns ones. The code handles temporal length from the actual latent tensor rather than assuming a fixed 49-frame input.

## Weighted drift objective

For predicted and real clean latents, form offset-$\tau$ differences and spatially average each channel to get drift vectors $\hat\mu_{b,i}$ and $\mu^*_{b,i}$. The existing relative pair loss is

$$
\ell_{b,i}=\frac{\|\hat\mu_{b,i}-\mu^*_{b,i}\|_2^2}
{\operatorname{stopgrad}(\max(\|\mu^*_{b,i}\|_2^2,10^{-8}))}.
$$

PhyStrata changes only its aggregation:

$$
L_{\mathrm{drift}}=0.4\,\underbrace{\operatorname{mean}_b[(1-\sigma_b^2)_+]}_{\text{existing noise suppression}}
\operatorname{mean}_{b,i}[\operatorname{stopgrad}(W_bU_{b,i})\ell_{b,i}].
$$

The complete denoiser objective is the original denoising loss plus this auxiliary term. The predictor continues to use its existing separate training objective. No parameter shape or inference setting is changed. Setting both $\beta_v$ and $\beta_f$ to zero recovers the original drift aggregation; their upper bounds give $W_bU_{b,i}\in[0.36,1.96]$.

## Boundaries of the interpretation

- High latent activity can reflect camera motion, occlusion, texture changes, or flicker. It is a supervision allocation signal, not a physical-law verdict.
- The frame credit is derived from intervals touching a latent position; it is not a causal attribution of motion to an individual frame.
- The bound applies to the additional multiplier. It does not bound the entire gradient norm when the original relative-loss denominator is small.
