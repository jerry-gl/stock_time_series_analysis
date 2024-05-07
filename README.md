# Overview

The project encompasses three key components:

1. Forecasting the future value of the US corporate profit margin after tax.
2. Use the forecasted value to analyse the relationship between the US market index and selected major economic indicators.
3. Develop a model aimed at evaluate the US market sentiment dynamics.

This analysis aims to shed light on how the market index interacts with specific economic indicators, blending time series analysis with fundamental economic principles.

## Data (as of March 31, 2024)
![Selected Economic Indicators](/pictures/economic_indicators.png)

These selected metrics are significant economic indicators that have the potential to influence the overall stock market:

1. "market_index" denotes the Wilshire5000 index, a comprehensive measure of the total U.S. stock market's performance.
2. "gdp" signifies the nominal gross domestic product (GDP) of the United States, a key indicator of the nation's economic health and productivity.
3. "interest_rate" represents the US federal funds rate, expressed as a percentage, which serves as a pivotal benchmark for short-term interest rates and business cost of borrowing.
4. "cpi" stands for the Consumer Price Index (CPI) for All Urban Consumers: All Items in U.S. City Average, a widely used measure of inflation that reflects changes in the cost of living over time.
5. "profit_margin" indicates the US corporate profit after tax rate, expressed as a percentage, offering insights into how technology advancements have increased business's profitability.

The "date" column represents the date beginning in that quarter. For example, "2024-01-01" indicates the date range from January 1, 2024, to March 31, 2024.

## Forecasting Profit Margin
![In Sample Predictions vs Observerd Profit Margin Values](/pictures/profit_margin_in_samples.png)
![Profit Margin Forecasts](/pictures/profit_margin_forecasts.JPG)

The decision has been made to adopt the ARIMA model for profit margin forecasting, owing to its demonstrated superiority in minimising Mean Absolute Percentage Error (MAPE) compared to alternative models, thereby prioritising forecasting accuracy. 

The simplicity and straightforward implementation of the ARIMA model further contribute to its favourable selection for forecasting needs. Notably, the ARIMA(0,1,0) specification implies that the profit margin behaves akin to a random walk with a drift component, indicating a linear trend in the profit margin, which aligns well with established economic principles.

## Regression Modelling

![Regression Models](/pictures/regression_models.png)

$$ \text{Market Framework} = 44.4095 + 37.07265 \times \text{gdp} + 18.30313 \times \text{interest rate} + 21.99363 \times \text{profit margin} $$


## Market Sentiment
![Market Sentiment](/pictures/market_sentiment.png)

$$ \text{Market Sentiment} = -0.0025 -0.0093 \times \text{interest rate} + 0.0821 \times \text{profit margin} $$

The standard deviation from the mean within the Market Sentiment model serves as a potential metric for identifying notable shifts in investor sentiment toward the market. Instances where values significantly diverge from the mean signal potential extremes in market perception, indicating conditions that may be characterised as either excessively bullish ("hot") or overly bearish ("cold"). 

These deviations provide insight into the degree of divergence from typical expectations, thereby facilitating the recognition of significant shifts in market sentiment.

### Credits

The data used in this analysis project was obtained from the following sources:

1. [WILL5000IND](https://fred.stlouisfed.org/series/WILL5000IND) - Wilshire 5000 Total Market Full Cap Index, retrieved from the Federal Reserve Economic Data (FRED) database.
2. [GDP](https://fred.stlouisfed.org/series/GDP#0) - Gross Domestic Product, retrieved from the Federal Reserve Economic Data (FRED) database.
3. [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) - Effective Federal Funds Rate, retrieved from the Federal Reserve Economic Data (FRED) database.
4. [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL#0) - Consumer Price Index for All Urban Consumers: All Items, retrieved from the Federal Reserve Economic Data (FRED) database.
5. [Corporate Profit Margin After Tax](https://www.gurufocus.com/economic_indicators/62/corporate-profit-margin-after-tax-) - Corporate profit margin after tax, retrieved from GuruFocus.

This project is inspired by previous collaboration with Leo Wang.

### Disclaimers
The information provided in this analysis is for educational and informational purposes only. It is not intended as investment advice, and should not be construed as such. Investing in financial markets involves risk, including the risk of losing principal. Past performance is not indicative of future results. The forecasts, models, and analyses presented here are based on historical data and assumptions that may not reflect future market conditions accurately. Before making any investment decisions, it is essential to conduct thorough research and consider consulting with a qualified financial advisor who can assess your individual financial situation and goals. The author of this analysis and the platform providing it shall not be liable for any losses or damages arising from the use of this information.
