import pandas as pd
import numpy as np
import os
from config.system_config import DATA_DIRS


def create_sales_data():
    """
    Create sample sales data and save to CSV.
    """
    # Create sample directory if it doesn't exist
    sample_dir = DATA_DIRS["sample_datasets"]
    os.makedirs(sample_dir, exist_ok=True)

    # Define the file path
    sample_file = os.path.join(sample_dir, "sales_data.csv")

    # Create sample data
    np.random.seed(42)
    n_rows = 1000

    regions = ['North', 'South', 'East', 'West', 'Central']
    product_categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Office']

    data = {
        'date': pd.date_range(start='2023-01-01', periods=n_rows).astype(str),
        'product_id': np.random.randint(1000, 9999, size=n_rows),
        'product_category': np.random.choice(product_categories, size=n_rows),
        'region': np.random.choice(regions, size=n_rows),
        'price': np.round(np.random.uniform(10, 1000, size=n_rows), 2),
        'quantity': np.random.randint(1, 50, size=n_rows),
        'customer_id': np.random.randint(10000, 99999, size=n_rows)
    }

    # Introduce some missing values
    for col in ['price', 'quantity', 'region']:
        mask = np.random.choice([True, False], size=n_rows, p=[0.05, 0.95])
        data[col] = pd.Series(data[col])
        data[col][mask] = None

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(sample_file, index=False)
    print(f"Sample sales data created and saved to {sample_file}")

    return sample_file


def create_customer_data():
    """
    Create sample customer data and save to CSV.
    """
    # Create sample directory if it doesn't exist
    sample_dir = DATA_DIRS["sample_datasets"]
    os.makedirs(sample_dir, exist_ok=True)

    # Define the file path
    sample_file = os.path.join(sample_dir, "customer_data.csv")

    # Create sample data
    np.random.seed(42)
    n_rows = 1000

    segments = ['Premium', 'Standard', 'Basic']
    countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan']

    today = pd.to_datetime('2023-12-31')

    data = {
        'customer_id': np.random.randint(10000, 99999, size=n_rows),
        'age': np.random.randint(18, 80, size=n_rows),
        'customer_segment': np.random.choice(segments, size=n_rows, p=[0.2, 0.5, 0.3]),
        'country': np.random.choice(countries, size=n_rows),
        'registration_date': [(today - pd.Timedelta(days=np.random.randint(1, 1500))).strftime('%Y-%m-%d') for _ in
                              range(n_rows)],
        'membership_years': np.random.randint(0, 10, size=n_rows),
        'purchase_amount': np.round(np.random.exponential(scale=500, size=n_rows), 2),
        'service_fee': np.round(np.random.uniform(10, 100, size=n_rows), 2),
        'rating': np.round(np.random.uniform(1, 5, size=n_rows), 1)
    }

    # Introduce some missing values
    for col in ['age', 'purchase_amount', 'rating']:
        mask = np.random.choice([True, False], size=n_rows, p=[0.05, 0.95])
        data[col] = pd.Series(data[col])
        data[col][mask] = None

    # Introduce a smaller number of missing values for customer_id
    mask = np.random.choice([True, False], size=n_rows, p=[0.01, 0.99])
    data['customer_id'] = pd.Series(data['customer_id'])
    data['customer_id'][mask] = None

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(sample_file, index=False)
    print(f"Sample customer data created and saved to {sample_file}")

    return sample_file