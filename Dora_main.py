from Dora_EDA import MyModel

model = MyModel()
model.load_data(filepath="training_set_VU_DM.csv")
high_corrs = model.get_highly_correlated_columns(threshold=0.7)

for col1, col2, corr in high_corrs:
    print(f"High Correlation:{col1} vs {col2}: correlation = {corr:.3f}")

high_missing_cols = model.get_columns_with_high_missing(threshold=0.8)
model.drop_columns(high_missing_cols)
model.drop_columns(["booking_bool"])
print(len(f"Num high missing cols: {high_missing_cols})"))


print("New dataframe after dropping high-missing columns:")
print(model.dataframe)

model.summarize_numerical_columns()
model.get_non_numerical_columns()
model.get_columns_with_missing_values()