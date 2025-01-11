-- Main backup database in postgresql
CREATE DATABASE gml_ghg;

-- Table schema for cleaned NOAA GML data 
CREATE TABLE "CO2DataNOAA" (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP WITH TIME ZONE,
    site VARCHAR(50),
    ppm FLOAT,
    latitude FLOAT,
    longitude FLOAT,
    altitude FLOAT,
    elevation FLOAT,
    intake_height FLOAT,
    qcflag VARCHAR(50),
    year INTEGER,
    month INTEGER,
    day INTEGER,
    season VARCHAR(20),
    co2_change_rate FLOAT,
    gas VARCHAR(10)
);

CREATE TABLE "CH4DataNOAA" (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP WITH TIME ZONE,
    site VARCHAR(50),
    ppm FLOAT,
    latitude FLOAT,
    longitude FLOAT,
    altitude FLOAT,
    elevation FLOAT,
    intake_height FLOAT,
    qcflag VARCHAR(50),
    year INTEGER,
    month INTEGER,
    day INTEGER,
    season VARCHAR(20),
    ch4_change_rate FLOAT,
    gas VARCHAR(10)
);

CREATE TABLE "N2ODataNOAA" (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP WITH TIME ZONE,
    site VARCHAR(50),
    ppm FLOAT,
    latitude FLOAT,
    longitude FLOAT,
    altitude FLOAT,
    elevation FLOAT,
    intake_height FLOAT,
    qcflag VARCHAR(50),
    year INTEGER,
    month INTEGER,
    day INTEGER,
    season VARCHAR(20),
    n2o_change_rate FLOAT,
    gas VARCHAR(10)
);

CREATE TABLE "SF6DataNOAA" (
    id SERIAL PRIMARY KEY,
    datetime TIMESTAMP WITH TIME ZONE,
    site VARCHAR(50),
    ppm FLOAT,
    latitude FLOAT,
    longitude FLOAT,
    altitude FLOAT,
    elevation FLOAT,
    intake_height FLOAT,
    qcflag VARCHAR(50),
    year INTEGER,
    month INTEGER,
    day INTEGER,
    season VARCHAR(20),
    sf6_change_rate FLOAT,
    gas VARCHAR(10)
);