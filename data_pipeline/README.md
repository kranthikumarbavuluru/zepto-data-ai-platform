# Module 1: Data Pipeline (/data_pipeline)

## Overview
This module automates the extraction, cleaning, currency conversion, relational storage, and querying of competitive catalog data from `books.toscrape.com`.

## Currency Conversion Baseline
As required by the assignment specification, currency enrichment is computed using the project-defined fixed baseline rate:
$$\text{price\_inr} = \text{price\_gbp} \times 105.50$$
This is a fixed constant requiring no external API or lookup.

## Cleaning Decisions
1. **Currency Parsing**: Stripped the leading `£` currency symbol using regex `[^\d.]` and cast the result to `float`.
2. **Missing / Corrupt Price Rows**: Any row with an unparseable price is dropped rather than imputed. In catalog and competitive pricing pipelines, price integrity is paramount; imputing catalog prices risks introducing artificial bias.
3. **Star Rating**: String ratings (`One` through `Five`) are converted to integer values ($1$–$5$). Any unmapped or missing rating defaults to the median value ($3$).
4. **Availability**: Converted to a boolean integer flag (`1` for in-stock, `0` otherwise).

## Relational Schema
Normalized into two tables to eliminate redundancy:
- `categories(category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE)`
- `books(book_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))`

## How to Run
```bash
# 1. Run scraping, cleaning, and database loading
python data_pipeline/pipeline.py

# 2. Execute SQL queries and pandas in-memory verification
python data_pipeline/queries.py
