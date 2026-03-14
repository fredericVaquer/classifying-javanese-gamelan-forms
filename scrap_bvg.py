import os
from pathlib import Path
from playwright.sync_api import sync_playwright

# List of target genres to categorize by
GENRE_LIST = ["Ladrang", "Ketawang", "Ayak", "Srepegan", "Sampak", "Bubaran", "Lancaran"]

def get_genre(title):
    for g in GENRE_LIST:
        if g.lower() in title.lower():
            return g
    return "Other_Gendhing"

def run_scraper():
    root_dir = Path("BVG_Categorized_Data")
    
    with sync_playwright() as p:
        # Launch browser in headed mode to help bypass simple bot-detection
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        
        print("Navigating to BVG Index...")
        page.goto("https://www.gamelanbvg.com/gendhing/index.php")
        
        # Wait for the table/links to appear
        page.wait_for_selector("a[href*='showGendhing']", timeout=60000)
        
        # Get all unique links
        links = page.eval_on_selector_all("a[href*='showGendhing']", "elements => elements.map(e => e.href)")
        links = list(set(links))
        print(f"Found {len(links)} songs. Starting download...")

        for link in links:
            try:
                page.goto(link)
                # Wait for title to load
                page.wait_for_selector("h2")
                title = page.inner_text("h2").strip().replace("/", "-")
                
                # Determine destination
                genre = get_genre(title)
                song_dir = root_dir / genre / title
                song_dir.mkdir(parents=True, exist_ok=True)
                
                # Save content
                content = page.content()
                with open(song_dir / f"{title}.html", "w", encoding="utf-8") as f:
                    f.write(content)
                
                print(f" [OK] {genre} -> {title}")
            except Exception as e:
                print(f" [ERROR] Could not save {link}: {e}")
        
        browser.close()

if __name__ == "__main__":
    run_scraper()