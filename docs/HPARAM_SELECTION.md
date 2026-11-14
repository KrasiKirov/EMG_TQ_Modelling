# Hyperparameter selection

All choices use session-1 validation loss (mean-squared error, purged validation split) only; session-2 test scores were not used. Every comparison is relative to the previous network (8 units × 2 layers, dropout 0.3, learning rate 0.003), person by person, so every person counts equally. Rule: the lowest mean log-ratio is the best; the chosen one is the smallest network within one standard error of it. All runs: window-study reference (200 ms, active torque, causal 2nd-order envelope, no shuffling, no per-angle scaling), batch 64.

## 1. Grid screen (150 epochs, 2 seeds × 6 people)

42 complete configurations of 42: units 8/16 (1 or 2 layers) and 32 (1 layer), dropout 0.1/0.3, learning rate 0.001/0.003/0.01, plus edge values for 8 and 16 units (dropout 0 with learning rate 0.01/0.03, dropout 0.1 with 0.03). Many runs were still improving at the 150-epoch limit, so this step only picks the finalists: the 4 best, the smallest network within one standard error of the best (marked), and the previous network.

| Configuration | Parameters | Validation loss vs previous network (150 epochs) | ± SE | People better |
| --- | --- | --- | --- | --- |
| `HT_u16_l2_d00_lr01` **best** | 3537 | -50.8 % | 18.2 % | 6/6 |
| `HT_u32_l1_d10_lr01` | 4897 | -45.9 % | 16.0 % | 6/6 |
| `HT_u32_l1_d10_lr003` | 4897 | -45.7 % | 13.3 % | 6/6 |
| `HT_u16_l2_d10_lr01` | 3537 | -45.1 % | 18.2 % | 6/6 |
| `HT_u16_l1_d10_lr01` | 1425 | -42.0 % | 16.4 % | 6/6 |
| `HT_u8_l2_d00_lr01` **chosen** | 1001 | -41.2 % | 20.3 % | 6/6 |
| `HT_u16_l1_d00_lr01` | 1425 | -41.1 % | 18.5 % | 6/6 |
| `HT_u32_l1_d30_lr003` | 4897 | -39.1 % | 13.8 % | 6/6 |
| `HT_u16_l1_d10_lr003` | 1425 | -38.0 % | 13.1 % | 6/6 |
| `HT_u16_l2_d10_lr003` | 3537 | -36.4 % | 21.8 % | 5/6 |
| `HT_u32_l1_d30_lr01` | 4897 | -35.9 % | 21.0 % | 5/6 |
| `HT_u16_l2_d30_lr003` | 3537 | -33.3 % | 20.6 % | 5/6 |
| `HT_u16_l2_d30_lr01` | 3537 | -30.7 % | 18.6 % | 5/6 |
| `HT_u8_l1_d10_lr01` | 457 | -30.2 % | 15.5 % | 6/6 |
| `HT_u8_l1_d00_lr01` | 457 | -28.4 % | 17.8 % | 5/6 |
| `HT_u32_l1_d10_lr001` | 4897 | -28.3 % | 14.3 % | 5/6 |
| `HT_u16_l1_d30_lr003` | 1425 | -27.6 % | 13.5 % | 6/6 |
| `HT_u16_l1_d00_lr03` | 1425 | -26.6 % | 23.3 % | 4/6 |
| `HT_u16_l1_d30_lr01` | 1425 | -25.4 % | 19.5 % | 5/6 |
| `HT_u16_l2_d10_lr001` | 3537 | -24.6 % | 21.0 % | 4/6 |
| `HT_u32_l1_d30_lr001` | 4897 | -24.2 % | 12.4 % | 5/6 |
| `HT_u8_l2_d10_lr001` | 1001 | -21.0 % | 15.7 % | 4/6 |
| `HT_u8_l2_d10_lr01` | 1001 | -20.6 % | 26.5 % | 4/6 |
| `HT_u16_l2_d30_lr001` | 3537 | -15.6 % | 20.1 % | 5/6 |
| `HT_u8_l1_d10_lr003` | 457 | -15.1 % | 17.5 % | 5/6 |
| `HT_u8_l2_d10_lr003` | 1001 | -13.8 % | 12.1 % | 5/6 |
| `HT_u8_l1_d30_lr01` | 457 | -11.7 % | 17.6 % | 3/6 |
| `HT_u16_l1_d10_lr001` | 1425 | -7.0 % | 21.0 % | 5/6 |
| `HT_u16_l1_d30_lr001` | 1425 | -7.0 % | 13.1 % | 2/6 |
| `HT_u8_l2_d30_lr001` | 1001 | -6.6 % | 14.9 % | 2/6 |
| `HT_u8_l2_d30_lr003` | 1001 | +0.0 % | 0.0 % | 0/6 |
| `HT_u8_l1_d30_lr003` | 457 | +1.7 % | 17.7 % | 1/6 |
| `HT_u8_l2_d10_lr03` | 1001 | +3.2 % | 20.5 % | 2/6 |
| `HT_u8_l2_d30_lr01` | 1001 | +3.4 % | 23.4 % | 2/6 |
| `HT_u16_l2_d10_lr03` | 3537 | +6.9 % | 24.4 % | 3/6 |
| `HT_u8_l1_d10_lr001` | 457 | +14.2 % | 16.6 % | 1/6 |
| `HT_u8_l1_d00_lr03` | 457 | +24.1 % | 11.7 % | 1/6 |
| `HT_u8_l1_d30_lr001` | 457 | +25.3 % | 16.9 % | 1/6 |
| `HT_u8_l1_d10_lr03` | 457 | +25.9 % | 18.3 % | 2/6 |
| `HT_u16_l1_d10_lr03` | 1425 | +27.2 % | 44.6 % | 3/6 |
| `HT_u8_l2_d00_lr03` | 1001 | +37.2 % | 18.2 % | 1/6 |
| `HT_u16_l2_d00_lr03` | 3537 | +143.0 % | 84.8 % | 3/6 |

## 2. Finalists (400 epochs, 3 seeds × 6 people)

| Configuration | Parameters | Validation loss vs previous network (400 epochs) | ± SE | People better |
| --- | --- | --- | --- | --- |
| `HF_u16_l2_d00_lr01` **best** **chosen** | 3537 | -42.4 % | 7.5 % | 6/6 |
| `HF_u32_l1_d10_lr003` | 4897 | -39.4 % | 3.8 % | 6/6 |
| `HF_u16_l2_d10_lr01` | 3537 | -36.6 % | 9.0 % | 6/6 |
| `HF_u8_l2_d00_lr01` | 1001 | -36.0 % | 10.5 % | 6/6 |
| `HF_u32_l1_d10_lr01` | 4897 | -34.9 % | 8.2 % | 6/6 |
| `HF_u8_l2_d30_lr003` | 1001 | +0.0 % | 0.0 % | 0/6 |

**Chosen:** `HF_u16_l2_d00_lr01`: units 16, layers 2, dropout 0.0, learning rate 0.01.

## 3. Epoch limit

The chosen network with up to 400 epochs against the same network with 150 (paired by person and seed, 12 pairs): validation loss -0.0 %, lower for 1/6 people. Rule: use 400 if that is at least 5 % lower. **The study uses at most 150 epochs.**
With the 400-epoch limit the chosen network stopped by itself (early stopping) after a median of 135 epochs and never reached 400; runs that hit 150 in the screen did so by chance of their trajectory (runs are not bit-reproducible), not because the limit was systematically too short.

## 4. Batch size (reported only)

`HB8_u16_l2_d00_lr01` (batch 8, original A0 training) against `HF_u16_l2_d00_lr01` (batch 64, 400 epochs), paired by person and seed (18 pairs): validation loss +280.9 %, lower for 0/6 people. The study keeps batch 64: about 5x faster per epoch.
Caveat: this is batch 8 at the learning rate tuned for batch 64 (0.01). Smaller batches give noisier updates and usually need a smaller learning rate; the batch-8 runs peaked early (median best epoch 53) and then stopped. So this shows that 0.01 is too high for batch 8, not that batch 8 is worse in general; batch 8 was not tuned.
