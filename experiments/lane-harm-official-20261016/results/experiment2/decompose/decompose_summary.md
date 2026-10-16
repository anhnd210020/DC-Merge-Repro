# Experiment 2b/2c: where EuroSAT's harm comes from, and the dose test (validation)

## dtd-eurosat-gtsrb-svhn: operator **official**, α = 0.4 (0.4)

EuroSAT accuracy: standalone 98.56, merged alone 61.11, own lane only (shared frame) 72.59, full context 61.19.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | -11.48 |
| dtd alone | 1.89 |
| gtsrb alone | -0.67 |
| svhn alone | 5.11 |
| interaction dtd+gtsrb | 1.59 |
| interaction dtd+svhn | 0.85 |
| interaction gtsrb+svhn | 3.81 |
| interaction dtd+gtsrb+svhn | -1.19 |
| **total harm vs merged alone** | **-0.07** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | dtd | gtsrb | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +8.59 | +0.80 | -1.62 | -15.44 | -1.92 |
| off: gtsrb | +3.56 | +2.02 | -16.67 | -4.96 | -4.01 |
| off: dtd | +3.15 | -4.63 | -0.08 | -1.27 | -0.71 |
| dtd=0.25 | +2.07 | -3.35 | +0.44 | -0.83 | -0.42 |
| dtd=0.5 | +1.70 | -2.50 | +0.20 | -0.58 | -0.29 |
| dtd=0.75 | +0.63 | -1.22 | +0.28 | -0.21 | -0.13 |
| gtsrb=0.25 | +2.37 | +1.60 | -13.70 | -3.59 | -3.33 |
| gtsrb=0.5 | +1.63 | +0.90 | -9.62 | -2.31 | -2.35 |
| gtsrb=0.75 | +1.15 | +0.43 | -4.35 | -1.09 | -0.97 |
| svhn=0.25 | +6.22 | +0.53 | -0.71 | -11.56 | -1.38 |
| svhn=0.5 | +3.78 | +0.37 | -0.16 | -7.61 | -0.90 |
| svhn=0.75 | +1.70 | +0.37 | +0.00 | -3.94 | -0.47 |
| gtsrb+svhn=0.25 | +8.96 | +1.97 | -13.66 | -15.25 | -4.49 |
| gtsrb+svhn=0.5 | +6.85 | +1.44 | -9.46 | -9.68 | -2.71 |
| gtsrb+svhn=0.75 | +2.93 | +0.64 | -4.39 | -5.01 | -1.46 |
| eurosat=1.25 | +4.74 | -0.21 | -0.12 | -0.25 | +1.04 |
| eurosat=1.5 | +9.96 | -0.59 | -0.79 | -0.61 | +1.99 |

## dtd-eurosat-gtsrb-svhn: operator **official**, α = 0.7 (selected)

EuroSAT accuracy: standalone 98.56, merged alone 67.41, own lane only (shared frame) 83.41, full context 59.33.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | -16.00 |
| dtd alone | 1.56 |
| gtsrb alone | -0.96 |
| svhn alone | 9.89 |
| interaction dtd+gtsrb | 2.78 |
| interaction dtd+svhn | 3.19 |
| interaction gtsrb+svhn | 7.00 |
| interaction dtd+gtsrb+svhn | 0.63 |
| **total harm vs merged alone** | **8.07** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | dtd | gtsrb | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +20.70 | +1.60 | -2.02 | -27.53 | -1.81 |
| off: gtsrb | +9.44 | +2.50 | -31.39 | -8.32 | -6.94 |
| off: dtd | +8.15 | -7.34 | +3.13 | -0.88 | +0.76 |
| dtd=0.25 | +6.26 | -5.96 | +2.93 | -0.63 | +0.65 |
| dtd=0.5 | +4.33 | -3.94 | +2.57 | -0.15 | +0.70 |
| dtd=0.75 | +2.33 | -2.29 | +1.90 | -0.13 | +0.45 |
| gtsrb=0.25 | +8.11 | +2.07 | -25.53 | -5.88 | -5.31 |
| gtsrb=0.5 | +6.41 | +1.49 | -18.37 | -3.27 | -3.43 |
| gtsrb=0.75 | +4.48 | +1.12 | -9.94 | -1.25 | -1.40 |
| svhn=0.25 | +16.26 | +1.33 | +0.12 | -20.61 | -0.73 |
| svhn=0.5 | +11.41 | +0.64 | +1.27 | -13.87 | -0.14 |
| svhn=0.75 | +6.37 | +0.21 | +1.31 | -6.65 | +0.31 |
| gtsrb+svhn=0.25 | +20.96 | +2.87 | -23.95 | -25.11 | -6.31 |
| gtsrb+svhn=0.5 | +15.96 | +2.07 | -16.43 | -16.33 | -3.68 |
| gtsrb+svhn=0.75 | +8.93 | +1.22 | -9.11 | -8.07 | -1.76 |
| eurosat=1.25 | +9.26 | -0.32 | -3.64 | -1.92 | +0.84 |
| eurosat=1.5 | +17.44 | -0.53 | -6.81 | -4.17 | +1.48 |

## dtd-eurosat-gtsrb-svhn: operator **rank**, α = 0.4 (0.4)

EuroSAT accuracy: standalone 98.56, merged alone 85.00, own lane only (shared frame) 82.41, full context 63.33.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | 2.59 |
| dtd alone | 1.48 |
| gtsrb alone | 0.15 |
| svhn alone | 5.70 |
| interaction dtd+gtsrb | 2.89 |
| interaction dtd+svhn | 3.78 |
| interaction gtsrb+svhn | 6.85 |
| interaction dtd+gtsrb+svhn | -1.78 |
| **total harm vs merged alone** | **21.67** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | dtd | gtsrb | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +14.56 | +0.85 | -3.44 | -25.85 | -3.47 |
| off: gtsrb | +8.11 | +1.44 | -33.65 | -7.16 | -7.82 |
| off: dtd | +6.37 | -10.37 | +2.85 | -0.56 | -0.43 |
| dtd=0.25 | +5.00 | -7.34 | +2.49 | -0.40 | -0.06 |
| dtd=0.5 | +2.96 | -4.84 | +1.35 | -0.25 | -0.20 |
| dtd=0.75 | +0.96 | -2.13 | +0.91 | +0.02 | -0.06 |
| gtsrb=0.25 | +6.74 | +1.28 | -27.32 | -5.32 | -6.15 |
| gtsrb=0.5 | +5.07 | +1.22 | -18.65 | -3.21 | -3.89 |
| gtsrb=0.75 | +3.00 | +0.59 | -9.90 | -1.36 | -1.92 |
| svhn=0.25 | +11.26 | +0.69 | -2.38 | -19.27 | -2.42 |
| svhn=0.5 | +8.19 | +0.69 | -0.79 | -12.75 | -1.17 |
| svhn=0.75 | +4.19 | +0.11 | +0.20 | -6.86 | -0.59 |
| gtsrb+svhn=0.25 | +16.37 | +1.22 | -27.67 | -22.80 | -8.22 |
| gtsrb+svhn=0.5 | +12.26 | +1.01 | -18.96 | -15.12 | -5.20 |
| gtsrb+svhn=0.75 | +7.30 | +0.69 | -9.86 | -8.01 | -2.47 |
| eurosat=1.25 | +8.93 | -0.05 | -1.78 | -1.34 | +1.44 |
| eurosat=1.5 | +16.22 | -0.85 | -3.33 | -2.80 | +2.31 |

## dtd-eurosat-gtsrb-svhn: operator **rank**, α = 0.5 (selected)

EuroSAT accuracy: standalone 98.56, merged alone 89.41, own lane only (shared frame) 86.96, full context 60.96.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | 2.44 |
| dtd alone | 1.52 |
| gtsrb alone | 0.52 |
| svhn alone | 6.30 |
| interaction dtd+gtsrb | 3.74 |
| interaction dtd+svhn | 5.52 |
| interaction gtsrb+svhn | 9.89 |
| interaction dtd+gtsrb+svhn | -1.48 |
| **total harm vs merged alone** | **28.44** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | dtd | gtsrb | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +20.22 | +1.01 | -1.31 | -31.16 | -2.81 |
| off: gtsrb | +12.67 | +1.70 | -42.91 | -8.76 | -9.33 |
| off: dtd | +9.30 | -12.77 | +4.32 | -0.36 | +0.12 |
| dtd=0.25 | +7.22 | -9.63 | +3.84 | -0.10 | +0.33 |
| dtd=0.5 | +4.85 | -5.74 | +2.81 | +0.25 | +0.54 |
| dtd=0.75 | +2.59 | -2.66 | +1.46 | +0.35 | +0.44 |
| gtsrb=0.25 | +11.44 | +1.12 | -34.52 | -6.57 | -7.13 |
| gtsrb=0.5 | +9.44 | +0.96 | -24.43 | -3.65 | -4.42 |
| gtsrb=0.75 | +5.89 | +0.74 | -11.88 | -1.04 | -1.57 |
| svhn=0.25 | +16.93 | +0.48 | -0.36 | -23.30 | -1.56 |
| svhn=0.5 | +12.22 | +0.21 | +0.79 | -15.33 | -0.53 |
| svhn=0.75 | +6.78 | +0.48 | +0.99 | -7.74 | +0.13 |
| gtsrb+svhn=0.25 | +22.59 | +1.97 | -32.38 | -27.78 | -8.90 |
| gtsrb+svhn=0.5 | +17.85 | +1.12 | -22.80 | -18.17 | -5.50 |
| gtsrb+svhn=0.75 | +11.44 | +0.80 | -10.57 | -8.82 | -1.79 |
| eurosat=1.25 | +11.22 | -0.37 | -3.17 | -2.17 | +1.38 |
| eurosat=1.5 | +19.52 | -1.49 | -6.41 | -4.74 | +1.72 |

## eurosat-gtsrb-mnist-svhn: operator **official**, α = 0.4 (0.4)

EuroSAT accuracy: standalone 98.56, merged alone 61.11, own lane only (shared frame) 72.93, full context 63.59.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | -11.81 |
| gtsrb alone | -0.74 |
| mnist alone | -0.11 |
| svhn alone | 3.81 |
| interaction gtsrb+mnist | 1.93 |
| interaction gtsrb+svhn | 4.04 |
| interaction mnist+svhn | 1.67 |
| interaction gtsrb+mnist+svhn | -1.26 |
| **total harm vs merged alone** | **-2.48** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | mnist | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +8.26 | -1.31 | -4.25 | -15.14 | -3.11 |
| off: mnist | +2.22 | +0.83 | -18.20 | -0.04 | -3.80 |
| off: gtsrb | +3.96 | -15.12 | +1.95 | -5.42 | -3.66 |
| gtsrb=0.25 | +3.33 | -11.76 | +1.55 | -3.71 | -2.65 |
| gtsrb=0.5 | +2.15 | -8.12 | +0.85 | -2.27 | -1.85 |
| gtsrb=0.75 | +1.30 | -4.20 | +0.30 | -1.04 | -0.91 |
| mnist=0.25 | +2.11 | +0.99 | -14.90 | +0.19 | -2.90 |
| mnist=0.5 | +1.22 | +0.87 | -10.65 | +0.31 | -2.06 |
| mnist=0.75 | +0.70 | +0.91 | -5.40 | +0.29 | -0.87 |
| svhn=0.25 | +6.41 | -0.83 | -3.10 | -11.62 | -2.29 |
| svhn=0.5 | +4.33 | -0.59 | -2.00 | -7.66 | -1.48 |
| svhn=0.75 | +2.15 | -0.16 | -1.25 | -3.76 | -0.76 |
| gtsrb+svhn=0.25 | +8.70 | -12.47 | -2.70 | -14.50 | -5.24 |
| gtsrb+svhn=0.5 | +6.48 | -8.04 | -1.75 | -9.72 | -3.26 |
| gtsrb+svhn=0.75 | +3.44 | -4.43 | -0.90 | -4.94 | -1.71 |
| eurosat=1.25 | +5.63 | +0.24 | +0.15 | -0.29 | +1.43 |
| eurosat=1.5 | +10.48 | -0.20 | +0.35 | -0.81 | +2.46 |

## eurosat-gtsrb-mnist-svhn: operator **official**, α = 0.8 (selected)

EuroSAT accuracy: standalone 98.56, merged alone 68.85, own lane only (shared frame) 86.59, full context 59.44.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | -17.74 |
| gtsrb alone | -0.22 |
| mnist alone | -0.00 |
| svhn alone | 8.19 |
| interaction gtsrb+mnist | 1.93 |
| interaction gtsrb+svhn | 9.52 |
| interaction mnist+svhn | 5.11 |
| interaction gtsrb+mnist+svhn | 2.63 |
| **total harm vs merged alone** | **9.41** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | mnist | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +25.44 | +2.41 | -8.70 | -29.79 | -2.66 |
| off: mnist | +9.67 | +7.44 | -31.70 | -1.34 | -3.98 |
| off: gtsrb | +13.85 | -34.84 | +2.45 | -8.68 | -6.80 |
| gtsrb=0.25 | +12.74 | -29.22 | +2.45 | -5.57 | -4.90 |
| gtsrb=0.5 | +11.56 | -21.54 | +2.10 | -2.94 | -2.70 |
| gtsrb=0.75 | +7.11 | -11.92 | +1.00 | -0.85 | -1.16 |
| mnist=0.25 | +7.52 | +6.45 | -24.75 | -0.69 | -2.87 |
| mnist=0.5 | +5.33 | +5.70 | -15.30 | -0.33 | -1.15 |
| mnist=0.75 | +2.78 | +3.60 | -6.80 | -0.04 | -0.11 |
| svhn=0.25 | +21.41 | +3.40 | -6.55 | -24.07 | -1.45 |
| svhn=0.5 | +16.07 | +3.92 | -4.05 | -15.89 | +0.01 |
| svhn=0.75 | +8.89 | +2.69 | -1.65 | -7.99 | +0.49 |
| gtsrb+svhn=0.25 | +26.00 | -24.23 | -3.50 | -28.12 | -7.46 |
| gtsrb+svhn=0.5 | +21.19 | -17.06 | -2.30 | -18.40 | -4.14 |
| gtsrb+svhn=0.75 | +14.63 | -8.95 | -1.00 | -8.53 | -0.96 |
| eurosat=1.25 | +11.07 | -5.19 | -0.70 | -2.92 | +0.57 |
| eurosat=1.5 | +18.52 | -10.49 | -0.95 | -6.34 | +0.18 |

## eurosat-gtsrb-mnist-svhn: operator **rank**, α = 0.4 (0.4)

EuroSAT accuracy: standalone 98.56, merged alone 85.00, own lane only (shared frame) 82.63, full context 64.33.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | 2.37 |
| gtsrb alone | 0.07 |
| mnist alone | 0.81 |
| svhn alone | 4.59 |
| interaction gtsrb+mnist | 3.15 |
| interaction gtsrb+svhn | 6.56 |
| interaction mnist+svhn | 3.41 |
| interaction gtsrb+mnist+svhn | -0.30 |
| **total harm vs merged alone** | **20.67** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | mnist | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +14.26 | -2.06 | -8.00 | -25.57 | -5.34 |
| off: mnist | +7.07 | +4.43 | -27.20 | -1.13 | -4.21 |
| off: gtsrb | +9.48 | -31.91 | +1.00 | -6.88 | -7.08 |
| gtsrb=0.25 | +7.85 | -25.22 | +0.80 | -5.03 | -5.40 |
| gtsrb=0.5 | +6.19 | -18.53 | +0.45 | -3.07 | -3.74 |
| gtsrb=0.75 | +3.59 | -9.82 | -0.05 | -1.50 | -1.94 |
| mnist=0.25 | +5.63 | +3.92 | -19.00 | -0.58 | -2.51 |
| mnist=0.5 | +3.96 | +3.33 | -12.15 | -0.02 | -1.22 |
| mnist=0.75 | +2.33 | +2.02 | -5.65 | +0.13 | -0.29 |
| svhn=0.25 | +12.04 | -0.16 | -5.45 | -18.69 | -3.07 |
| svhn=0.5 | +8.04 | +0.51 | -3.60 | -12.50 | -1.89 |
| svhn=0.75 | +4.89 | +0.79 | -1.85 | -6.32 | -0.62 |
| gtsrb+svhn=0.25 | +16.22 | -24.19 | -4.10 | -22.97 | -8.76 |
| gtsrb+svhn=0.5 | +12.96 | -17.89 | -2.35 | -15.21 | -5.62 |
| gtsrb+svhn=0.75 | +7.37 | -9.46 | -1.05 | -7.55 | -2.67 |
| eurosat=1.25 | +8.63 | -1.94 | -0.65 | -1.86 | +1.04 |
| eurosat=1.5 | +15.56 | -3.92 | -1.40 | -3.65 | +1.65 |

## eurosat-gtsrb-mnist-svhn: operator **rank**, α = 0.6 (selected)

EuroSAT accuracy: standalone 98.56, merged alone 93.30, own lane only (shared frame) 90.22, full context 53.04.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | 3.07 |
| gtsrb alone | 1.15 |
| mnist alone | 1.41 |
| svhn alone | 5.19 |
| interaction gtsrb+mnist | 4.59 |
| interaction gtsrb+svhn | 16.15 |
| interaction mnist+svhn | 6.52 |
| interaction gtsrb+mnist+svhn | 2.19 |
| **total harm vs merged alone** | **40.26** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | mnist | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +30.04 | +4.04 | -5.25 | -34.34 | -1.38 |
| off: mnist | +14.70 | +11.44 | -28.95 | -1.33 | -1.03 |
| off: gtsrb | +24.07 | -46.00 | +1.60 | -11.08 | -7.85 |
| gtsrb=0.25 | +21.70 | -37.93 | +1.60 | -7.01 | -5.41 |
| gtsrb=0.5 | +17.63 | -28.27 | +1.35 | -3.42 | -3.18 |
| gtsrb=0.75 | +10.89 | -14.33 | +0.80 | -0.86 | -0.88 |
| mnist=0.25 | +11.81 | +9.98 | -18.85 | -0.12 | +0.71 |
| mnist=0.5 | +8.00 | +7.05 | -10.10 | +0.23 | +1.29 |
| mnist=0.75 | +3.70 | +4.20 | -4.35 | +0.15 | +0.93 |
| svhn=0.25 | +24.63 | +4.99 | -3.05 | -26.10 | +0.12 |
| svhn=0.5 | +18.30 | +4.71 | -1.70 | -17.04 | +1.07 |
| svhn=0.75 | +10.30 | +2.57 | -0.75 | -8.22 | +0.97 |
| gtsrb+svhn=0.25 | +34.63 | -32.54 | +0.00 | -30.98 | -7.22 |
| gtsrb+svhn=0.5 | +29.41 | -22.09 | +0.25 | -20.02 | -3.11 |
| gtsrb+svhn=0.75 | +19.19 | -11.64 | +0.15 | -9.95 | -0.56 |
| eurosat=1.25 | +12.00 | -2.89 | -0.70 | -3.71 | +1.18 |
| eurosat=1.5 | +22.70 | -7.56 | -1.20 | -7.36 | +1.65 |

## eurosat-gtsrb-sun397-svhn: operator **official**, α = 0.4 (0.4)

EuroSAT accuracy: standalone 98.56, merged alone 61.11, own lane only (shared frame) 72.63, full context 63.33.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | -11.52 |
| gtsrb alone | -0.93 |
| sun397 alone | -0.19 |
| svhn alone | 4.81 |
| interaction gtsrb+sun397 | 0.44 |
| interaction gtsrb+svhn | 4.48 |
| interaction sun397+svhn | 1.41 |
| interaction gtsrb+sun397+svhn | -0.74 |
| **total harm vs merged alone** | **-2.22** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | sun397 | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +9.96 | -2.02 | -0.13 | -15.96 | -2.04 |
| off: sun397 | +0.93 | +0.59 | -2.12 | +0.08 | -0.13 |
| off: gtsrb | +3.26 | -15.48 | -0.25 | -4.76 | -4.31 |
| gtsrb=0.25 | +2.74 | -12.79 | -0.15 | -3.40 | -3.40 |
| gtsrb=0.5 | +2.30 | -9.14 | -0.05 | -1.92 | -2.20 |
| gtsrb=0.75 | +1.07 | -5.15 | -0.03 | -0.92 | -1.25 |
| sun397=0.25 | +0.81 | +0.24 | -1.91 | +0.15 | -0.18 |
| sun397=0.5 | +0.70 | +0.32 | -1.21 | -0.02 | -0.05 |
| sun397=0.75 | +0.30 | +0.20 | -0.43 | -0.08 | -0.00 |
| svhn=0.25 | +8.19 | -1.27 | +0.13 | -11.91 | -1.22 |
| svhn=0.5 | +5.22 | -1.11 | +0.03 | -7.68 | -0.89 |
| svhn=0.75 | +2.63 | -0.55 | +0.00 | -4.00 | -0.48 |
| gtsrb+svhn=0.25 | +9.00 | -13.30 | +0.20 | -14.96 | -4.77 |
| gtsrb+svhn=0.5 | +7.26 | -9.54 | +0.03 | -9.70 | -2.99 |
| gtsrb+svhn=0.75 | +3.70 | -5.38 | +0.03 | -5.03 | -1.67 |
| eurosat=1.25 | +5.56 | -0.36 | -0.15 | -0.23 | +1.20 |
| eurosat=1.5 | +10.96 | -1.19 | -0.25 | -0.60 | +2.23 |

## eurosat-gtsrb-sun397-svhn: operator **official**, α = 0.8 (selected)

EuroSAT accuracy: standalone 98.56, merged alone 68.85, own lane only (shared frame) 85.96, full context 62.78.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | -17.11 |
| gtsrb alone | -0.63 |
| sun397 alone | -1.59 |
| svhn alone | 11.37 |
| interaction gtsrb+sun397 | 1.93 |
| interaction gtsrb+svhn | 10.70 |
| interaction sun397+svhn | 2.04 |
| interaction gtsrb+sun397+svhn | -0.63 |
| **total harm vs merged alone** | **6.07** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | sun397 | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +23.48 | +3.72 | +0.35 | -30.73 | -0.79 |
| off: sun397 | +1.74 | +8.59 | -3.93 | +1.77 | +2.04 |
| off: gtsrb | +11.37 | -31.27 | +0.53 | -7.84 | -6.80 |
| gtsrb=0.25 | +10.37 | -26.80 | -0.03 | -5.03 | -5.37 |
| gtsrb=0.5 | +8.70 | -19.36 | -0.05 | -3.11 | -3.45 |
| gtsrb=0.75 | +5.37 | -10.97 | -0.10 | -1.25 | -1.74 |
| sun397=0.25 | +1.22 | +7.09 | -2.97 | +1.27 | +1.65 |
| sun397=0.5 | +0.78 | +5.27 | -1.69 | +0.73 | +1.27 |
| sun397=0.75 | +0.33 | +2.81 | -0.98 | +0.48 | +0.66 |
| svhn=0.25 | +21.41 | +5.30 | +0.25 | -23.57 | +0.85 |
| svhn=0.5 | +15.81 | +5.54 | +0.18 | -15.41 | +1.53 |
| svhn=0.75 | +8.93 | +3.44 | +0.15 | -7.78 | +1.19 |
| gtsrb+svhn=0.25 | +24.26 | -21.97 | +0.53 | -26.78 | -5.99 |
| gtsrb+svhn=0.5 | +21.30 | -14.77 | +0.38 | -17.25 | -2.59 |
| gtsrb+svhn=0.75 | +13.33 | -8.27 | +0.08 | -8.61 | -0.87 |
| eurosat=1.25 | +10.41 | -4.55 | -0.58 | -2.73 | +0.64 |
| eurosat=1.5 | +18.33 | -8.79 | -1.56 | -5.88 | +0.53 |

## eurosat-gtsrb-sun397-svhn: operator **rank**, α = 0.4 (0.4)

EuroSAT accuracy: standalone 98.56, merged alone 85.00, own lane only (shared frame) 82.85, full context 69.22.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | 2.15 |
| gtsrb alone | 0.78 |
| sun397 alone | -0.74 |
| svhn alone | 6.04 |
| interaction gtsrb+sun397 | -0.33 |
| interaction gtsrb+svhn | 6.48 |
| interaction sun397+svhn | 0.22 |
| interaction gtsrb+sun397+svhn | 1.19 |
| **total harm vs merged alone** | **15.78** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | sun397 | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +13.93 | -4.28 | -0.10 | -25.82 | -4.07 |
| off: sun397 | +0.33 | +0.44 | -4.74 | +0.52 | -0.86 |
| off: gtsrb | +8.11 | -33.57 | +0.33 | -5.90 | -7.76 |
| gtsrb=0.25 | +6.74 | -27.20 | +0.35 | -4.05 | -6.04 |
| gtsrb=0.5 | +5.63 | -19.56 | +0.05 | -2.23 | -4.03 |
| gtsrb=0.75 | +3.22 | -9.86 | -0.20 | -0.86 | -1.93 |
| sun397=0.25 | +0.37 | +0.67 | -3.50 | +0.54 | -0.48 |
| sun397=0.5 | +0.48 | +0.44 | -2.34 | +0.44 | -0.25 |
| sun397=0.75 | -0.30 | +0.40 | -1.36 | +0.27 | -0.25 |
| svhn=0.25 | +11.63 | -2.06 | -0.40 | -19.48 | -2.58 |
| svhn=0.5 | +7.96 | -0.95 | -0.30 | -12.52 | -1.45 |
| svhn=0.75 | +4.70 | -0.12 | -0.18 | -6.45 | -0.51 |
| gtsrb+svhn=0.25 | +13.96 | -27.36 | +0.53 | -22.26 | -8.78 |
| gtsrb+svhn=0.5 | +11.81 | -19.24 | +0.40 | -14.58 | -5.40 |
| gtsrb+svhn=0.75 | +7.04 | -9.86 | -0.03 | -7.32 | -2.54 |
| eurosat=1.25 | +8.19 | -2.41 | -0.50 | -1.13 | +1.03 |
| eurosat=1.5 | +14.85 | -3.96 | -0.83 | -2.55 | +1.88 |

## eurosat-gtsrb-sun397-svhn: operator **rank**, α = 0.6 (selected)

EuroSAT accuracy: standalone 98.56, merged alone 93.30, own lane only (shared frame) 90.70, full context 61.19.

**Decomposition of harm vs merged-alone** (pp; terms add up exactly):

| part | pp |
|---|---:|
| frame cost (merged alone → own lane in shared frame) | 2.59 |
| gtsrb alone | 1.22 |
| sun397 alone | -1.22 |
| svhn alone | 7.04 |
| interaction gtsrb+sun397 | 2.07 |
| interaction gtsrb+svhn | 18.22 |
| interaction sun397+svhn | 0.89 |
| interaction gtsrb+sun397+svhn | 1.30 |
| **total harm vs merged alone** | **32.11** |

**Dose test** (change vs full context, pp):

| setting | EuroSAT | gtsrb | sun397 | svhn | mean of 4 |
|---|---:|---:|---:|---:|---:|
| off: svhn | +27.44 | +3.01 | +1.13 | -34.54 | -0.74 |
| off: sun397 | +3.04 | +4.71 | -5.19 | +2.00 | +1.14 |
| off: gtsrb | +22.81 | -49.21 | +2.17 | -8.22 | -8.11 |
| gtsrb=0.25 | +20.33 | -40.70 | +1.74 | -5.90 | -6.13 |
| gtsrb=0.5 | +17.33 | -30.29 | +1.26 | -3.07 | -3.69 |
| gtsrb=0.75 | +10.85 | -15.28 | +0.76 | -0.44 | -1.03 |
| sun397=0.25 | +2.30 | +4.24 | -3.40 | +1.48 | +1.15 |
| sun397=0.5 | +1.41 | +3.17 | -1.61 | +1.11 | +1.02 |
| sun397=0.75 | +0.74 | +1.86 | -1.01 | +0.54 | +0.53 |
| svhn=0.25 | +25.33 | +4.67 | +1.26 | -25.99 | +1.32 |
| svhn=0.5 | +19.26 | +4.59 | +0.81 | -17.04 | +1.90 |
| svhn=0.75 | +11.11 | +3.60 | +0.50 | -7.84 | +1.85 |
| gtsrb+svhn=0.25 | +30.22 | -34.64 | +2.34 | -28.43 | -7.63 |
| gtsrb+svhn=0.5 | +27.85 | -23.40 | +1.84 | -17.75 | -2.86 |
| gtsrb+svhn=0.75 | +19.30 | -10.61 | +1.06 | -8.18 | +0.39 |
| eurosat=1.25 | +13.00 | -4.47 | -0.65 | -2.69 | +1.30 |
| eurosat=1.5 | +22.26 | -8.95 | -1.16 | -5.57 | +1.65 |

