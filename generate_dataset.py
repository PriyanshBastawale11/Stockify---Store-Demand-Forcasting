import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_dataset(n_products_per_store=20, start_date=datetime(2020, 1, 1), end_date=None):  
    """
    Generate a retail dataset with historical time series data.
    
    Parameters:
    - n_products_per_store: Number of unique products per store
    - start_date: Starting date for the dataset (defaults to January 1, 2020)
    - end_date: Ending date for the dataset (defaults to current date)
    
    Returns:
    - DataFrame with historical retail data
    """
    if end_date is None:
        end_date = datetime.now()
    
    # Calculate the number of days in the date range
    days_history = (end_date - start_date).days + 1
    
    stores = {
        'STORE001': {'area': 'Nagpur East', 'is_urban': True, 'income_level': 'high', 
                    'size_factor': 1.2, 'base_footfall': 80},  
        'STORE002': {'area': 'Nagpur West', 'is_urban': True, 'income_level': 'medium',
                    'size_factor': 1.0, 'base_footfall': 65},  
        'STORE003': {'area': 'Nagpur North', 'is_urban': False, 'income_level': 'medium',
                    'size_factor': 0.8, 'base_footfall': 50},  
        'STORE004': {'area': 'Nagpur South', 'is_urban': True, 'income_level': 'high',
                    'size_factor': 1.1, 'base_footfall': 75},  
        'STORE005': {'area': 'Nagpur Central', 'is_urban': True, 'income_level': 'medium',
                    'size_factor': 1.0, 'base_footfall': 70},  
    }
    
    categories = {
        'Groceries': {
            'subcategories': ['Staples', 'Snacks', 'Beverages'],
            'price_range': (20, 200),  
            'margin_range': (8, 15),    
            'base_demand': (10, 30),
            'trend_factor': 0.05  # Annual growth trend
        },
        'Fresh Produce': {
            'subcategories': ['Fruits', 'Vegetables', 'Dairy'],
            'price_range': (30, 150),
            'margin_range': (12, 20),
            'base_demand': (15, 35),
            'trend_factor': 0.08  # Annual growth trend
        },
        'Personal Care': {
            'subcategories': ['Hygiene', 'Cosmetics', 'Healthcare'],
            'price_range': (50, 300),
            'margin_range': (15, 25),
            'base_demand': (5, 15),
            'trend_factor': 0.10  # Annual growth trend
        },
        'Home Essentials': {
            'subcategories': ['Cleaning', 'Kitchen', 'Storage'],
            'price_range': (40, 250),
            'margin_range': (10, 20),
            'base_demand': (8, 20),
            'trend_factor': 0.07  # Annual growth trend
        }
    }
    
    # Create product catalog first
    products = []
    product_id_counter = 1
    
    for store_id in stores.keys():
        for _ in range(n_products_per_store):
            category = random.choice(list(categories.keys()))
            category_info = categories[category]
            subcategory = random.choice(category_info['subcategories'])
            
            product_id = f"P{product_id_counter:04d}"
            product_id_counter += 1
            
            base_price = random.uniform(*category_info['price_range'])
            margin = random.uniform(*category_info['margin_range'])
            
            products.append({
                'store_id': store_id,
                'product_id': product_id,
                'category': category,
                'subcategory': subcategory,
                'base_price': base_price,
                'margin': margin
            })
    
    # Generate time series data for each product
    records = []
    
    # Pre-calculate festival dates for each year
    festival_dates = {}
    for year in range(start_date.year, end_date.year + 1):
        # Approximate festival dates (month, day)
        festival_dates[year] = [
            (datetime(year, 10, 15), 1.3),  # Dussehra
            (datetime(year, 11, 4), 1.2),   # Diwali
            (datetime(year, 12, 25), 1.2),  # Christmas
            (datetime(year, 8, 15), 1.1),   # Independence Day
            (datetime(year, 3, 21), 1.1)    # Holi (approximate)
        ]
    
    # Generate daily data for each product
    for product in products:
        store_id = product['store_id']
        store_info = stores[store_id]
        category = product['category']
        category_info = categories[category]
        
        # Initialize time-varying parameters
        current_price = product['base_price']
        current_margin = product['margin']
        
        # Base values that will evolve over time
        base_demand = random.uniform(*category_info['base_demand'])
        
        # Generate data for each day
        for day in range(days_history):
            current_date = start_date + timedelta(days=day)
            
            # Time-based trend (annual growth)
            time_factor = 1 + (category_info['trend_factor'] * day / 365)
            
            # Seasonal pattern (monthly)
            month = current_date.month
            seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * month / 12)
            
            # Weekly pattern (weekends have higher demand)
            weekday = current_date.weekday()
            weekend_boost = 1.2 if weekday >= 5 else 1.0  # Weekend boost
            
            # Festival effects
            festival_boost = 1.0
            for fest_date, boost in festival_dates.get(current_date.year, []):
                # Apply festival effect for 7 days around the festival
                if abs((current_date - fest_date).days) <= 7:
                    festival_boost = max(festival_boost, boost * (1 - abs((current_date - fest_date).days) / 10))
            
            # Occasional price changes (every 30-60 days on average)
            if random.random() < 0.02:  # ~2% chance of price change each day
                price_change_factor = random.uniform(0.95, 1.05)
                current_price = current_price * price_change_factor
            
            # Occasional promotions
            is_promotional = random.random() < 0.1  # 10% chance of being a promotional day
            discount = random.randint(5, 20) if is_promotional else random.randint(0, 5)
            
            # Calculate demand with all factors
            actual_demand = int(base_demand * 
                              time_factor * 
                              seasonal_factor * 
                              weekend_boost *
                              festival_boost * 
                              store_info['size_factor'] *
                              (1 + 0.3 * (discount / 100)))  # Discount effect on demand
            
            # Add some noise to demand
            actual_demand = max(1, int(actual_demand * random.uniform(0.9, 1.1)))
            
            # Calculate daily revenue
            effective_price = current_price * (1 - discount/100)
            daily_revenue = actual_demand * effective_price
            
            # Stock levels (influenced by previous days, but simplified here)
            stock = random.randint(
                int(actual_demand * 5),  
                int(actual_demand * 15)   
            )
            
            # Customer footfall varies by day
            base_footfall = store_info['base_footfall']
            footfall_variation = random.uniform(0.8, 1.2) * weekend_boost * festival_boost
            customer_footfall = int(base_footfall * footfall_variation * seasonal_factor)
            
            # Credit limit calculation
            base_credit = 50000  
            credit_multiplier = {
                'high': 1.5,
                'medium': 1.0,
                'low': 0.8
            }[store_info['income_level']]
            credit_limit = int(base_credit * credit_multiplier * store_info['size_factor'])
            
            # Competitor price (follows our price with some variation)
            competitor_price = current_price * random.uniform(0.92, 1.08)
            
            record = {
                'date': current_date,
                'store_id': store_id,
                'area': store_info['area'],
                'is_urban': store_info['is_urban'],
                'income_level': store_info['income_level'],
                'store_size': 'Large' if store_info['size_factor'] > 1.1 else 
                            'Small' if store_info['size_factor'] < 0.9 else 'Medium',
                'storage_capacity': int(1000 * store_info['size_factor']),
                'product_id': product['product_id'],
                'category': category,
                'subcategory': product['subcategory'],
                'base_price': round(product['base_price'], 2),
                'current_price': round(current_price, 2),
                'margin': round(current_margin, 2),
                'discount': discount,
                'is_promotional_period': is_promotional,
                'credit_limit': credit_limit,
                'customer_footfall': customer_footfall,
                'stock': stock,
                'seasonal_factor': round(seasonal_factor, 3),
                'weekend_factor': weekend_boost,
                'festival_boost': round(festival_boost, 3),
                'trend_factor': round(time_factor, 3),
                'actual_demand': actual_demand,
                'daily_revenue': round(daily_revenue, 2),
                'stock_days_remaining': max(1, min(30, int(stock / max(1, actual_demand)))),
                'competitor_price': round(competitor_price, 2),
                'day_of_week': weekday,
                'month': month,
                'year': current_date.year
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values('date')
    return df

if __name__ == "__main__":
    print("Generating historical dataset...")
    # Generate data from January 1, 2020 to current date
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    # You can customize the parameters here
    df = generate_dataset(n_products_per_store=20, start_date=start_date, end_date=end_date)
    df.to_csv("retail_data_historical.csv", index=False)
    print(f"Dataset saved to retail_data_historical.csv with {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique products: {df['product_id'].nunique()}")
    print(f"Number of unique stores: {df['store_id'].nunique()}")
    print(f"Number of days in dataset: {(df['date'].max() - df['date'].min()).days + 1}")
