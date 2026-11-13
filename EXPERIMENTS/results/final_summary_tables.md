## Per subject and direction: mean over the 2 toes-down positions (seed-mean; R² floored at −1 per position)

| subject | direction | R² baseline | R² final | MSE baseline (Nm²) | MSE final | RMSE baseline (Nm) | RMSE final | NRMSE baseline | NRMSE final |
|---|---|---|---|---|---|---|---|---|---|
| YES | learned S1 → tested S2 | 0.05 | **-0.06** | 0.89 | 0.46 | 0.84 | 0.67 | 0.198 | 0.193 |
| YES | learned S2 → tested S1 | 0.63 | **0.24** | 0.55 | 0.57 | 0.70 | 0.75 | 0.127 | 0.153 |
| EG | learned S1 → tested S2 | 0.26 | **0.96** | 3.93 | 0.33 | 1.92 | 0.55 | 0.282 | 0.073 |
| EG | learned S2 → tested S1 | 0.67 | **0.96** | 1.79 | 0.22 | 1.27 | 0.46 | 0.175 | 0.063 |
| JM | learned S1 → tested S2 | 0.09 | **0.84** | 0.42 | 0.08 | 0.58 | 0.27 | 0.234 | 0.110 |
| JM | learned S2 → tested S1 | 0.20 | **0.84** | 0.44 | 0.09 | 0.65 | 0.30 | 0.272 | 0.124 |
| IES | learned S1 → tested S2 | 0.64 | **0.96** | 3.26 | 0.42 | 1.75 | 0.63 | 0.196 | 0.068 |
| IES | learned S2 → tested S1 | 0.89 | **0.97** | 0.95 | 0.31 | 0.96 | 0.54 | 0.100 | 0.054 |
| CH | learned S1 → tested S2 | 0.96 | **0.98** | 1.50 | 0.82 | 1.19 | 0.86 | 0.079 | 0.054 |
| CH | learned S2 → tested S1 | 0.96 | **0.98** | 1.69 | 0.87 | 1.23 | 0.92 | 0.074 | 0.056 |
| HM | learned S1 → tested S2 | 0.84 | **0.92** | 0.23 | 0.16 | 0.46 | 0.37 | 0.086 | 0.070 |
| HM | learned S2 → tested S1 | 0.93 | **0.93** | 0.13 | 0.13 | 0.33 | 0.35 | 0.064 | 0.066 |

## Groups (seed-mean per position, then mean)

| group | direction | positions | R² baseline | R² final | RMSE baseline (Nm) | RMSE final | positions improved (R²) |
|---|---|---|---|---|---|---|---|
| screening (YES, EG) | S1→S2 | toes-down | 0.156 | 0.453 | 1.38 | 0.61 | 3/4 |
| screening (YES, EG) | S1→S2 | other | 0.963 | 0.981 | 1.62 | 1.31 | 10/12 |
| screening (YES, EG) | S2→S1 | toes-down | 0.649 | 0.600 | 0.98 | 0.60 | 3/4 |
| screening (YES, EG) | S2→S1 | other | 0.907 | 0.978 | 1.71 | 1.42 | 6/12 |
| validation (JM, IES, CH, HM) | S1→S2 | toes-down | 0.632 | 0.926 | 1.00 | 0.54 | 8/8 |
| validation (JM, IES, CH, HM) | S1→S2 | other | 0.821 | 0.959 | 1.97 | 1.13 | 21/24 |
| validation (JM, IES, CH, HM) | S2→S1 | toes-down | 0.746 | 0.929 | 0.79 | 0.52 | 6/8 |
| validation (JM, IES, CH, HM) | S2→S1 | other | 0.813 | 0.941 | 2.16 | 1.39 | 16/24 |
| all 6 | S1→S2 | toes-down | 0.473 | 0.768 | 1.13 | 0.56 | 11/12 |
| all 6 | S1→S2 | other | 0.869 | 0.966 | 1.85 | 1.19 | 31/36 |
| all 6 | S2→S1 | toes-down | 0.714 | 0.819 | 0.86 | 0.55 | 9/12 |
| all 6 | S2→S1 | other | 0.844 | 0.954 | 2.01 | 1.40 | 22/36 |
