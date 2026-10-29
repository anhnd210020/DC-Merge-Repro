# Experiment 8: depth summary

Operators: ta, iso_cts, tsvm. Pairs: 38. Victims (8): sun397, stanford-cars, resisc45, eurosat, svhn, gtsrb, mnist, dtd. Smallest attainable p: 0.0039. Feasible (>= 5 victims): True.

## Confirmatory claims (fixed sequence, alpha = 0.05; victims as units, exact sign-flip tests)

| Order | Claim | Contrast | n | T (mean z_v) | 95% CI | p | Claim p (IUT) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | localization | phi_M-phi_E | 8 | 0.022 | [-0.049, 0.086] | 0.2617 | 0.2617 | not confirmed |
|  |  | phi_M-phi_Lt | 8 | 0.303 | [0.250, 0.360] | 0.0039 |  |  |
| 2 | specificity | e_M-e_E | 8 | -0.046 | [-0.116, 0.021] | 0.8945 | 0.8945 | not tested (sequence stopped earlier); exploratory |
|  |  | e_M-e_Lt | 8 | 0.176 | [0.122, 0.234] | 0.0039 |  |  |
| 3 | information | lam_M-lam_E | 8 | -0.259 | [-0.686, -0.020] | 0.9883 | 0.9883 | not tested (sequence stopped earlier); exploratory |
|  |  | lam_M-lam_Lt | 8 | -0.279 | [-0.584, 0.066] | 0.9336 |  |  |
| 4 | amplification | kappa_M | 8 | 0.471 | [0.398, 0.552] | 0.0039 | 0.0039 | not tested (sequence stopped earlier); exploratory |
| 5 | repair | repair | 8 | 0.283 | [0.206, 0.357] | 0.0039 | 0.0039 | not tested (sequence stopped earlier); exploratory |

## Effect sizes (pooled, ratios of sums; two-stage cluster-bootstrap 95% intervals)

| Quantity | Estimate | 95% CI |
| --- | --- | --- |
| depth share sigma_E | 0.458 | [0.404, 0.520] |
| depth share sigma_M | 0.398 | [0.370, 0.427] |
| depth share sigma_Lt | 0.144 | [0.098, 0.179] |
| amplification share kappa_M / Gamma | 0.423 | [0.338, 0.508] |
| victim recovery rho_v (mid removed) | 0.623 | [0.571, 0.671] |
| harmer cost rho_h (mid removed) | 0.407 | [0.369, 0.460] |
| information-loss fraction psi | 0.315 | [0.196, 0.454] |

## Per operator (estimates with intervals; not tested)

| Operator | pairs | sigma_E | sigma_M | sigma_Lt |
| --- | --- | --- | --- | --- |
| ta | 16 | 0.541 [0.473, 0.619] | 0.355 [0.312, 0.400] | 0.104 [0.055, 0.143] |
| iso_cts | 6 | 0.305 [0.112, 0.528] | 0.583 [0.319, 0.696] | 0.112 [0.014, 0.252] |
| tsvm | 16 | 0.307 [0.251, 0.359] | 0.452 [0.419, 0.496] | 0.241 [0.193, 0.299] |

## Sensitivity analyses (pre-declared; do not change verdicts)

- k = 1 (top context per victim): claim p-values {'localization': 0.34765625, 'specificity': 0.88671875, 'information': 0.98046875, 'amplification': 0.00390625, 'repair': 0.00390625}
- mean of per-pair ratios, sigma: {'E': 0.41102787989043216, 'M': 0.4377008367775051, 'Lt': 0.15127128333206258}
- specificity with one random seed: {'e_M-e_E': 0.91015625, 'e_M-e_Lt': 0.00390625}
- pre/post single-unit mean fractions: {'pre': {'removed': 0.059550625557903795, 'created': 0.029709868267453365}, 'post': {'removed': -0.0005423090860521006, 'created': -0.00017298769285111292}}
- leave one victim out (T, min, max, sign change):
    - phi_M-phi_E: 0.022, 0.006, 0.044, False
    - phi_M-phi_Lt: 0.303, 0.286, 0.321, False
    - e_M-e_E: -0.046, -0.064, -0.026, False
    - e_M-e_Lt: 0.176, 0.158, 0.195, False
    - lam_M-lam_E: -0.259, -0.302, -0.077, False
    - lam_M-lam_Lt: -0.279, -0.411, -0.202, False
    - kappa_M: 0.471, 0.438, 0.490, False
    - repair: 0.283, 0.262, 0.300, False

## Exploratory (no p-values)

- harmers' own-skill share by third: {'E': 0.09056754261995079, 'M': 0.4890195371498227, 'Lt': 0.4204129202302266}
- victim's own-update share by third: {'E': -0.09570567657562681, 'M': 0.4117832680472813, 'Lt': 0.6839224085283455}
- repair by removed third (rho_v, rho_h): {'E': {'rho_v': 0.633681987221499, 'rho_h': 0.12847682250056136}, 'M': {'rho_v': 0.6226167725930073, 'rho_h': 0.40719542368320905}, 'Lt': {'rho_v': 0.2543411581588442, 'rho_h': 0.29206355711891774}}
- attention vs MLP (fraction of Gamma): {'E:attn': {'removed': 0.12549101801532306, 'created': 0.0033455711462501464}, 'E:mlp': {'removed': 0.5260518584187528, 'created': 0.17819886939351884}, 'M:attn': {'removed': 0.23566264197311654, 'created': -0.0018977142673734248}, 'M:mlp': {'removed': 0.5420883572341624, 'created': 0.13060596974559346}, 'Lt:attn': {'removed': 0.13442921257348148, 'created': 0.010088182943425615}, 'Lt:mlp': {'removed': 0.18064541754375218, 'created': 0.02687243969575296}}
- block removed fraction: {'pre': 0.10899974131541912, '0': 0.17414908221843836, '1': 0.14913709656518098, '2': 0.2176915193207528, '3': 0.2498049884797123, '4': 0.25970729297695944, '5': 0.26523436808999307, '6': 0.22432883698201048, '7': 0.17094774418756, '8': 0.12531850315073023, '9': 0.08879927773810053, '10': 0.08972083198389838, '11': 0.04804558744100109, 'post': -0.0005395210264551963}
- block created fraction: {'pre': 0.04076372277161671, '0': 0.04026334620678135, '1': 0.02521411161621225, '2': 0.0365437950617063, '3': 0.014424335989040667, '4': 0.02115087075840652, '5': 0.033503398585306106, '6': 0.01839485656657666, '7': 0.010039201744108692, '8': 0.0035006342403614437, '9': 0.004306535752512232, '10': 0.004467811834911214, '11': 0.014678367396566989, 'post': -3.7104803787710106e-05}
- activation patching restored share by block: [0.19242506171886894, 0.2859501315059344, 0.4369576199943284, 0.5649606019543209, 0.7027769083502565, 0.8304489822837894, 0.9140086559203815, 0.9500477549665618, 0.9613712341003816, 0.9808659342067496, 1.0004780093755818, 1.0008622756369288]
- cross-term transport gain by block: [0.8190425471648389, 0.9095144859811194, 0.9503254956275836, 0.9268210510853476, 0.953678921596116, 0.9444287555119674, 1.0113033274161927, 1.045916227398469, 1.3358792104526056, 1.4255023706692462, 1.26139519294233, 1.0485031413825685]
- weight space: mean Spearman(STI, removed) 0.385; STI peaks [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; removal peaks [3, 2, 1, 2, 5, 2, 0, 3, 0, 0, 0, 0, 4, 2, 3, 3, 4, 4, 5, 5, 4, 2, 6, 5, 5, 4, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 6, 6]

## Who is responsible (secondary family, Holm)

- W1 interactions exist (two-sided): n=8, T=0.463, p=0.0078, Holm (0.0078125, True)
- W2 harm exceeds fake neighbours: n=8, T=1.123, p=0.0039, Holm (0.0078125, True)
- interaction share (ratio of sums): 0.461; fake/real harm: -0.117; cross-operator Shapley Spearman: 0.722 (n=9)
