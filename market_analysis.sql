-- Market Sentiment Analysis 
-- The objective is to present the US market index, nominal gdp, federal fund rate, and corporate profit margin all to one table on a quarterly basis for anaylsis.

CREATE SCHEMA `market_sentiment_analysis`;
USE market_sentiment_analysis;


/*
5 raw datas are imported into the database from the `source_data`:
	1. wilshire5000.csv
    2. gdp.csv
    3. interest_rate.csv
    4. cpi.csv
    5. profit_margin.csv
*/

-- Set column names and column datatypes.
RENAME TABLE `wilshire5000` TO `market_index`;
ALTER TABLE `market_index` 
CHANGE COLUMN `DATE` `date` DATE NULL DEFAULT NULL,
CHANGE COLUMN `WILL5000IND` `market_index` DOUBLE;

ALTER TABLE `gdp` 
CHANGE COLUMN `DATE` `date` DATE NULL DEFAULT NULL,
CHANGE COLUMN `GDPC1` `gdp` DOUBLE;

ALTER TABLE `interest_rate`
CHANGE COLUMN `DATE` `date` DATE NULL DEFAULT NULL,
CHANGE COLUMN `FEDFUNDS` `interest_rate` DOUBLE;

UPDATE `interest_rate`
SET `interest_rate` = `interest_rate` / 100; -- converts from percentage to decimals

ALTER TABLE `cpi`
CHANGE COLUMN `DATE` `date` DATE NULL DEFAULT NULL,
CHANGE COLUMN `CPIAUCSL` `cpi` DOUBLE;

ALTER TABLE `profit_margin` 
CHANGE COLUMN `DATE` `date` DATE NULL DEFAULT NULL,
CHANGE COLUMN `Corporate Profit Margin` `profit_margin` DOUBLE; 

UPDATE `profit_margin`
SET `profit_margin` = `profit_margin` / 100; -- converts from percentage to decimals

-- join all tables by the `market_index` table's `date` column.
SELECT m.date,
       m.market_index,
       g.gdp,
       i.interest_rate,
       c.cpi,
       p.profit_margin
FROM market_index m
LEFT JOIN gdp g ON m.date = g.date
LEFT JOIN interest_rate i ON m.date = i.date
LEFT JOIN cpi c ON m.date = c.date
LEFT JOIN profit_margin p ON m.date = p.date;





