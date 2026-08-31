from fetch_real_data import refresh_dataset


if __name__ == "__main__":
    df, metadata = refresh_dataset()
    print("Weekly refresh complete.")
    print(f"Rows in latest SQL view: {len(df)}")
    print(f"Snapshots stored: {metadata.get('snapshots', 0)}")
    print(f"Latest checked gameweek: {metadata.get('latest_checked_gameweek', 'Unknown')}")
