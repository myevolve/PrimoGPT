# Statistical Analysis of Trading Strategies

## ANOVA Test Results

One-way ANOVA test was performed to determine if there are statistically significant differences between strategy returns.

**ANOVA p-value:** 0.959876

The ANOVA test does not provide sufficient evidence of statistically significant differences between the performance of different strategies (p >= 0.05).

### ANOVA Table

```
               sum_sq     df         F    PR(>F)
C(strategy)  0.000132    5.0  0.206323  0.959876
Residual     0.108481  850.0       NaN       NaN
```

## Strategy Performance Summary

| Strategy     | Mean Return (%)   | Std Dev (%)   | CI Range (%)        |   Sample Size |
|:-------------|:------------------|:--------------|:--------------------|--------------:|
| Momentum     | 0.1398%           | 1.0222%       | [-0.0298%, 0.3093%] |           143 |
| PrimoRL      | 0.1344%           | 0.9622%       | [-0.0257%, 0.2946%] |           142 |
| MACD         | 0.0777%           | 0.8078%       | [-0.0563%, 0.2117%] |           143 |
| Price MA     | 0.0655%           | 0.9174%       | [-0.0867%, 0.2177%] |           143 |
| Buy and Hold | 0.0593%           | 1.6491%       | [-0.2143%, 0.3328%] |           143 |
| FinRL        | 0.0330%           | 1.1909%       | [-0.1653%, 0.2313%] |           142 |

## Confidence Intervals

<img src="confidence_intervals_plot.png" alt="Mean Returns with 95% Confidence Intervals by Strategy" style="max-width: 1000px; width: 100%;" />

## Pairwise Comparisons with PrimoRL

| PrimoRL vs   |   Mean Difference | % Difference   |   t-statistic |   p-value | PrimoRL Better?   | Statistically Significant?   |
|:-------------|------------------:|:---------------|--------------:|----------:|:------------------|:-----------------------------|
| FinRL        |       0.00101439  | 307.29%        |      0.786742 |  0.432123 | Yes               | No                           |
| Buy and Hold |       0.000751971 | 126.91%        |      0.468909 |  0.639581 | Yes               | No                           |
| MACD         |       0.000567292 | 72.99%         |      0.536968 |  0.591725 | Yes               | No                           |
| Momentum     |      -5.32192e-05 | -3.81%         |     -0.045099 |  0.96406  | No                | No                           |
| Price MA     |       0.000689168 | 105.16%        |      0.616575 |  0.538012 | Yes               | No                           |

## Statistical Significance Summary

PrimoRL outperforms 4 out of 5 other strategies

PrimoRL significantly outperforms 0 out of 5 other strategies (p < 0.05)
