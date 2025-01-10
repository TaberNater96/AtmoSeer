/*
SQL script to handle data transition from DynamoDB to Pandas DataFrame. This will be called using pandasql. Here, 
there are a few feature engineering steps that could be useful for EDA and preprocessing. These features are not meant
for final model input, they are designed for deeper insight for the EDA phase.
*/

SELECT 
    datetime, site, ppm, latitude, longitude, altitude, elevation,
    intake_height, qcflag, year, month, day, season, co2_change_rate,

    -- Rolling average of ppm for the last 10 entries including the current one, partitioned by site and year
    AVG(ppm) OVER (
        PARTITION BY site, year
        ORDER BY datetime 
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_ppm,
    
    -- Minimum ppm value for each site and year, calculated from the beginning of the year to the current row
    MIN(ppm) OVER (
        PARTITION BY site, year
        ORDER BY datetime
    ) AS min_ppm_year,
    
    -- Maximum ppm value for each site and year, calculated from the beginning of the year to the current row
    MAX(ppm) OVER (
        PARTITION BY site, year
        ORDER BY datetime
    ) AS max_ppm_year,

    -- Total count of ppm measurements for each site and year
    COUNT(ppm) OVER (
        PARTITION BY site, year
    ) AS total_site_ppm_annual,

    gas

FROM data_table
ORDER BY datetime, site;