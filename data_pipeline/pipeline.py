"""
Zepto Competitive Catalog Data Pipeline
Extract: Scrapes books.toscrape.com across multiple categories with pagination.
Transform: Cleans fields, validates types, and enriches with fixed-rate INR conversion (1 GBP = 105.50 INR).
Load: Normalizes and writes records into a relational SQLite database.
"""

import re
import sqlite3
from urllib.parse import urljoin
import bs4
import pandas as pd
import requests

BASE_URL = "http://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50  # Required fixed baseline conversion constant

# Mapping from text rating to integer 1-5
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

# Target categories: Travel, Mystery, Historical Fiction, Sequential Art (> 140 books total)
TARGET_CATEGORIES = [
    ("Travel", "catalogue/category/books/travel_2/index.html"),
    ("Mystery", "catalogue/category/books/mystery_3/index.html"),
    ("Historical Fiction", "catalogue/category/books/historical-fiction_4/index.html"),
    ("Sequential Art", "catalogue/category/books/sequential-art_5/index.html")
]

def scrape_category(category_name: str, relative_url: str):
    """Scrapes all books within a given category, automatically following pagination."""
    current_url = urljoin(BASE_URL, relative_url)
    books = []

    while current_url:
        resp = requests.get(current_url, timeout=15)
        resp.raise_for_status()
        soup = bs4.BeautifulSoup(resp.text, "html.parser")

        product_pods = soup.find_all("article", class_="product_pod")
        for pod in product_pods:
            # 1. Title
            title_anchor = pod.find("h3").find("a")
            raw_title = title_anchor.get("title") or title_anchor.text.strip()

            # 2. Price (raw with £)
            price_p = pod.find("p", class_="price_color")
            raw_price = price_p.text.strip() if price_p else ""

            # 3. Star rating (class name)
            rating_p = pod.find("p", class_=re.compile(r"star-rating\s+.*"))
            raw_rating = "Three"
            if rating_p:
                for cls in rating_p.get("class", []):
                    if cls in RATING_MAP:
                        raw_rating = cls
                        break

            # 4. Availability
            avail_p = pod.find("p", class_="availability")
            raw_avail = avail_p.text.strip() if avail_p else ""

            books.append({
                "raw_title": raw_title,
                "raw_price": raw_price,
                "raw_rating": raw_rating,
                "raw_avail": raw_avail,
                "category": category_name
            })

        # Check for 'next' page button
        next_li = soup.find("li", class_="next")
        if next_li and next_li.find("a"):
            next_href = next_li.find("a")["href"]
            current_url = urljoin(current_url, next_href)
        else:
            current_url = None

    print(f"Scraped {len(books)} books from category: '{category_name}'")
    return books

def clean_data(raw_records):
    """Cleans types, computes INR price, and enforces data constraints."""
    cleaned = []
    dropped_count = 0

    for r in raw_records:
        title = r["raw_title"].strip()
        if not title:
            dropped_count += 1
            continue

        # Strip currency symbol and parse float
        clean_price_str = re.sub(r"[^\d.]", "", r["raw_price"])
        try:
            price_gbp = float(clean_price_str)
        except (ValueError, TypeError):
            # Dropping corrupt pricing rows to protect downstream financial queries
            dropped_count += 1
            continue

        # Compute fixed-rate INR price (1 GBP = 105.50 INR)
        price_inr = round(price_gbp * GBP_TO_INR_RATE, 2)

        # Map rating to integer 1-5 (defaulting to median 3 if unmapped)
        rating = RATING_MAP.get(r["raw_rating"], 3)

        # Parse availability to boolean (1 for in-stock, 0 for out-of-stock)
        in_stock = 1 if "in stock" in r["raw_avail"].lower() else 0

        cleaned.append({
            "title": title,
            "price_gbp": price_gbp,
            "price_inr": price_inr,
            "rating": rating,
            "in_stock": in_stock,
            "category": r["category"]
        })

    print(f"Cleaning complete. Valid rows: {len(cleaned)}, Dropped rows: {dropped_count}")
    return pd.DataFrame(cleaned)

def init_db(db_path: str = "data_pipeline/zepto_catalog.db"):
    """Creates the SQLite tables according to schema.sql."""
    conn = sqlite3.connect(db_path)
    with open("data_pipeline/schema.sql", "r") as f:
        conn.executescript(f.read())
    return conn

def load_to_sqlite(df: pd.DataFrame, conn: sqlite3.Connection):
    """Loads cleaned data into categories and books tables."""
    cur = conn.cursor()

    # 1. Populate categories table
    unique_categories = df["category"].unique()
    for cat in unique_categories:
        cur.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()

    # 2. Get mapping of category_name -> category_id
    cur.execute("SELECT category_name, category_id FROM categories")
    cat_map = dict(cur.fetchall())

    # 3. Populate books table
    books_data = []
    for _, row in df.iterrows():
        books_data.append((
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            cat_map[row["category"]]
        ))

    cur.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?)
    """, books_data)
    conn.commit()
    print(f"Loaded {len(books_data)} books into 'books' table.")

def main():
    print("=== Starting Zepto Catalog Ingestion Pipeline ===")
    all_raw_books = []
    for cat_name, rel_url in TARGET_CATEGORIES:
        all_raw_books.extend(scrape_category(cat_name, rel_url))

    cleaned_df = clean_data(all_raw_books)
    conn = init_db("data_pipeline/zepto_catalog.db")
    load_to_sqlite(cleaned_df, conn)
    conn.close()
    print("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
