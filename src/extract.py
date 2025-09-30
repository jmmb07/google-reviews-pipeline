from dotenv import load_dotenv
import os
import requests
from pathlib import Path
import json
import csv
from time import sleep


load_dotenv()

API_KEY = os.getenv("SERPAPI_API_KEY")
PLACE_ID = os.getenv("PLACE_ID_GOOGLE")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_all_reviews():
    """Get all reviews with pagination"""
    all_reviews = []
    place_info = None
    next_token = None
    page = 1
    
    while True:
        print(f"Getting page {page}...")
        
        params = {
            "engine": "google_maps_reviews",
            "place_id": PLACE_ID,
            "hl": "pt-br",
            "api_key": API_KEY
        }
        
        if next_token:
            params["next_page_token"] = next_token
        
        try:
            r = requests.get("https://serpapi.com/search.json", params=params)
            r.raise_for_status()
            data = r.json()
            
            # store info
            if page == 1 and "place_info" in data:
                place_info = data["place_info"]
            
            # colect reviews
            if "reviews" in data:
                all_reviews.extend(data["reviews"])
                print(f"{len(data['reviews'])} collected reviews")
            
            # verify next page
            if "serpapi_pagination" in data and "next_page_token" in data["serpapi_pagination"]:
                next_token = data["serpapi_pagination"]["next_page_token"]
                page += 1
                sleep(1)  # pause
            else:
                print("\nAll pages collected!")
                break
                
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    return place_info, all_reviews

def extract_review_data(review):
    """Extract relevant data from one review"""
    return {
        "review_id":review.get("review_id", 0),
        "date": review.get("iso_date_of_last_edit", review.get("iso_date", "")),
        "user": review.get("user", {}).get("name", "Anônimo"),
        "rating": review.get("rating", 0),
        "comment": review.get("snippet", "")
    }

def save_data(place_info, reviews):
    """Saves collected data"""

    reviews_processadas = [extract_review_data(r) for r in reviews]

    loja_data = {
        "name": place_info["title"],
        "address": place_info.get("address", ""),
        "general_rating": place_info["rating"],
        "total_reviews_google": place_info["reviews"],
        "total_reviews_colected": len(reviews_processadas)
    }
    
    # save JSON from store
    with open(OUTPUT_DIR / "info_loja.json", "w", encoding="utf-8") as f:
        json.dump(loja_data, f, ensure_ascii=False, indent=2)
    
    # save reviews in JSON
    with open(OUTPUT_DIR / "reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews_processadas, f, ensure_ascii=False, indent=2)
    
    # save reviews in CSV
    with open(OUTPUT_DIR / "reviews.csv", "w", newline="", encoding="utf-8") as f:
        if reviews_processadas:
            writer = csv.DictWriter(f, fieldnames=reviews_processadas[0].keys())
            writer.writeheader()
            writer.writerows(reviews_processadas)
    
    # save JSON raw complete (backup)
    with open(OUTPUT_DIR / "raw_complete.json", "w", encoding="utf-8") as f:
        json.dump({
            "place_info": place_info,
            "reviews": reviews
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nFiles saved in '{OUTPUT_DIR}/':")
    print(f"   • info_store.json - Dados básicos da loja")
    print(f"   • reviews.json - Reviews processadas")
    print(f"   • reviews.csv - Reviews em CSV")
    print(f"   • raw_complete.json - Backup completo")

def main():
    print("Initiating collecting reviews...\n")
    
    # Busca reviews
    place_info, all_reviews = fetch_all_reviews()
    
    if not place_info or not all_reviews:
        print("No reviews collected!")
        return
    
    # Resumo
    print(f"\nCollection Summary:")
    print(f"   Store: {place_info['title']}")
    print(f"   Rating: {place_info['rating']} stars")
    print(f"   Reviews collected: {len(all_reviews)}")
    
    # Salva tudo
    save_data(place_info, all_reviews)
    
    print("\nCompleted!")


if __name__ == "__main__":
    main()