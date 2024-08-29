import pandas as pd
import time
from sklearn.model_selection import train_test_split

def load_datasets():
    start_time = time.time()
    events = pd.read_csv('data/events.csv')
    category_tree = pd.read_csv('data/category_tree.csv')
    item_properties_part1 = pd.read_csv('data/item_properties_part1.csv')
    item_properties_part2 = pd.read_csv('data/item_properties_part2.csv')
    return events, category_tree, item_properties_part1, item_properties_part2

def combine_item_properties(item_properties_part1, item_properties_part2):
    start_time = time.time()
    item_properties = pd.concat([item_properties_part1, item_properties_part2], ignore_index=True)
    return item_properties

def preprocess_item_properties(item_properties):
    start_time = time.time()
    item_properties.fillna('unknown', inplace=True)
    item_properties.drop_duplicates(inplace=True)
    item_properties.reset_index(drop=True, inplace=True)
    return item_properties

def extract_item_features(item_properties):
    start_time = time.time()
    item_features = item_properties.pivot_table(index='itemid', columns='property', values='value', aggfunc='first')
    item_features = item_features.reset_index()
    return item_features

def calculate_item_popularity(events):
    start_time = time.time()
    item_popularity = events.groupby('itemid').size().reset_index(name='popularity')
    item_event_type_counts = events.groupby(['itemid', 'event']).size().unstack(fill_value=0)
    item_popularity = item_popularity.merge(item_event_type_counts, on='itemid', how='left')
    return item_popularity

def combine_features(item_features, item_popularity):
    start_time = time.time()
    combined_features = item_features.merge(item_popularity, on='itemid', how='left')
    return combined_features

if __name__ == "__main__":
    start_time = time.time()
    
    events, category_tree, item_properties_part1, item_properties_part2 = load_datasets()
    item_properties = combine_item_properties(item_properties_part1, item_properties_part2)
    item_properties = preprocess_item_properties(item_properties)
    item_features = extract_item_features(item_properties)
    item_popularity = calculate_item_popularity(events)
    combined_item_data = combine_features(item_features, item_popularity)
    train_data, test_data = train_test_split(combined_item_data, test_size=0.2, random_state=42)