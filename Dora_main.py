from Dora_EDA import MyModel

model = MyModel()
model.load_data(filepath="training_set_VU_DM.csv")

#Finds columns that have a correlation value higher than 0.7
high_corrs = model.get_highly_correlated_columns(threshold=0.7)
for col1, col2, corr in high_corrs:
    print(f"High Correlation:{col1} vs {col2}: correlation = {corr:.3f}")

#Remove one of highly correlated columns
model.drop_columns(["booking_bool"])

#Finds and removes columns that have more than 80% missing values
high_missing_cols = model.get_columns_with_high_missing(threshold=0.8)
model.drop_columns(high_missing_cols)

#Prints the new dataset after dropping columns
print("New dataframe after dropping high-missing columns:")
print(model.dataframe)

#Prints out the summary of numericals columns such as range mean...
model.summarize_numerical_columns()
model.get_non_numerical_columns()

#Prints out columns that have missing values still
model.get_columns_with_missing_values()