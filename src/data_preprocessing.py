import pandas as pd
import time

def load_datasets():
    start_time = time.time()
    events = pd.read_csv('data/events.csv')
    category_tree = pd.read_csv('data/category_tree.csv')
    item_properties_part1 = pd.read_csv('data/item_properties_part1.csv')
    item_properties_part2 = pd.read_csv('data/item_properties_part2.csv')
    print(f"Datasets loaded in {time.time() - start_time:.2f} seconds")
    return events, category_tree, item_properties_part1, item_properties_part2

def combine_item_properties(item_properties_part1, item_properties_part2):
    start_time = time.time()
    item_properties = pd.concat([item_properties_part1, item_properties_part2], ignore_index=True)
    print(f"Item properties combined in {time.time() - start_time:.2f} seconds")
    return item_properties

def preprocess_item_properties(item_properties):
    start_time = time.time()
    item_properties.fillna('unknown', inplace=True)
    item_properties.drop_duplicates(inplace=True)
    item_properties.reset_index(drop=True, inplace=True)
    print(f"Item properties preprocessed in {time.time() - start_time:.2f} seconds")
    return item_properties

def extract_item_features(item_properties):
    start_time = time.time()
    item_features = item_properties.pivot_table(index='itemid', columns='property', values='value', aggfunc='first')
    item_features = item_features.reset_index()
    print(f"Item features extracted in {time.time() - start_time:.2f} seconds")
    return item_features

def calculate_item_popularity(events):
    start_time = time.time()
    item_popularity = events.groupby('itemid').size().reset_index(name='popularity')
    item_event_type_counts = events.groupby(['itemid', 'event']).size().unstack(fill_value=0)
    item_popularity = item_popularity.merge(item_event_type_counts, on='itemid', how='left')
    print(f"Item popularity calculated in {time.time() - start_time:.2f} seconds")
    return item_popularity

def combine_features(item_features, item_popularity):
    start_time = time.time()
    combined_features = item_features.merge(item_popularity, on='itemid', how='left')
    print(f"Features combined in {time.time() - start_time:.2f} seconds")
    return combined_features

if __name__ == "__main__":
    start_time = time.time()
    
    events, category_tree, item_properties_part1, item_properties_part2 = load_datasets()
    
    item_properties = combine_item_properties(item_properties_part1, item_properties_part2)
    
    print("Starting to preprocess item properties...")
    item_properties = preprocess_item_properties(item_properties)
    print("Item properties preprocessed.")
    
    print("Starting to extract item features...")
    item_features = extract_item_features(item_properties)
    print("Item features extracted.")
    
    print("Starting to calculate item popularity...")
    item_popularity = calculate_item_popularity(events)
    print("Item popularity calculated.")
    
    print("Starting to combine features...")
    combined_item_data = combine_features(item_features, item_popularity)
    print("Features combined.")
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(combined_item_data.head())