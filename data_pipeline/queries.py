"""
Zepto Competitive Catalog Analytics Queries
Demonstrates 5 required SQL clauses and pandas in-memory equivalence verification.
"""

import sqlite3
import pandas as pd

DB_PATH = "data_pipeline/zepto_catalog.db"

def execute_catalog_queries():
    conn = sqlite3.connect(DB_PATH)

    print("==================================================")
    print("QUERY 1: Filter Top-Rated In-Stock Books (SELECT / WHERE)")
    print("==================================================")
    q1 = """
    SELECT book_id, title, price_gbp, price_inr, rating
    FROM books
    WHERE rating = 5 AND in_stock = 1;
    """
    df_q1 = pd.read_sql(q1, conn)
    print(f"Total 5-star in-stock books: {len(df_q1)}")
    print(df_q1.head(5).to_string(index=False))

    print("\n==================================================")
    print("QUERY 2: Top 5 Most Expensive Books (ORDER BY / LIMIT)")
    print("==================================================")
    q2 = """
    SELECT book_id, title, price_inr, rating
    FROM books
    ORDER BY price_inr DESC
    LIMIT 5;
    """
    df_q2 = pd.read_sql(q2, conn)
    print(df_q2.to_string(index=False))

    print("\n==================================================")
    print("QUERY 3: Unique Ratings Represented in Catalog (DISTINCT)")
    print("==================================================")
    q3 = """
    SELECT DISTINCT rating
    FROM books
    ORDER BY rating ASC;
    """
    df_q3 = pd.read_sql(q3, conn)
    print(df_q3.to_string(index=False))

    print("\n==================================================")
    print("QUERY 4: Affordable Premium Books (BETWEEN)")
    print("==================================================")
    q4 = """
    SELECT book_id, title, price_gbp, price_inr, rating
    FROM books
    WHERE price_gbp BETWEEN 20.00 AND 35.00 AND rating >= 4
    ORDER BY price_gbp ASC
    LIMIT 5;
    """
    df_q4 = pd.read_sql(q4, conn)
    print(df_q4.to_string(index=False))

    print("\n==================================================")
    print("QUERY 5: Multi-Table Relationship (JOIN with Categories)")
    print("==================================================")
    q5 = """
    SELECT 
        b.book_id,
        b.title,
        c.category_name,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    WHERE b.rating >= 4
    ORDER BY b.price_inr DESC
    LIMIT 10;
    """
    df_q5_sql = pd.read_sql(q5, conn)
    print(df_q5_sql.to_string(index=False))

    print("\n==================================================")
    print("VERIFICATION: SQL JOIN vs. In-Memory Pandas pd.merge")
    print("==================================================")
    # 1. Fetch raw tables into pandas
    df_books = pd.read_sql("SELECT * FROM books", conn)
    df_categories = pd.read_sql("SELECT * FROM categories", conn)

    # 2. Replicate the JOIN, WHERE, ORDER BY, and LIMIT in Pandas
    df_merged = pd.merge(df_books, df_categories, on="category_id")
    df_filtered = df_merged[df_merged["rating"] >= 4]
    df_sorted = df_filtered.sort_values(by="price_inr", ascending=False).head(10)

    # 3. Align columns to match SQL output exactly
    df_q5_merge = df_sorted[[
        "book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"
    ]].reset_index(drop=True)

    print("\nPandas pd.merge output:")
    print(df_q5_merge.to_string(index=False))

    # 4. Check for equivalence
    is_identical = df_q5_sql.equals(df_q5_merge)
    print(f"\nAre SQL JOIN and pd.merge outputs identical? -> {is_identical}")

    conn.close()

if __name__ == "__main__":
    execute_catalog_queries()
