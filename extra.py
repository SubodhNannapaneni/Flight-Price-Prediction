
# %%
#As there is a huge variation between the economy and business class prices, trying
#to build seprate models for each class to reduce the error range.
sub_df.head()
# %%
sub_df['class'].value_counts()
# %%
bus_sub_df = sub_df.loc[sub_df['class'] == 1]
bus_sub_df.head()
# %%
df_preprocessed_bus = preprocessing(bus_sub_df)
# %%
df_preprocessed_bus.head()
# %%
Xb = df_preprocessed_bus.copy()
yb = Xb.pop("price")
# %%
##Prediction of the flight prices
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(Xb, yb, test_size = 0.25, random_state =1, shuffle = True)
# %%

scaler=StandardScaler()
x_train_scaled_bus=scaler.fit_transform(x_train_b)
x_test_scaled_bus=scaler.fit_transform(x_test_b)
x_train_scaled_bus=pd.DataFrame(x_train_scaled_bus, columns = x_train_b.columns)
x_test_scaled_bus=pd.DataFrame(x_test_scaled_bus, columns = x_test_b.columns)
# %%
x_train_scaled_bus.head()

# %%
logger.info("Start of Random Forest Regression for Business Class Data")

rfr_start_bus = time.time()
rfr_bus = RandomForestRegressor()
rfr_bus.fit(x_train_scaled_bus, y_train_b)
y_pred_rfr_bus = rfr_bus.predict(x_test_scaled_bus)

#Checking for the overfitting and the underfitting issues
logger.info('Training Accuracy {} '.format(r2_score(y_train_b,rfr_bus.predict(x_train_scaled_bus))*100))
logger.info('Validation Accuracy {} '.format(r2_score(y_test_b,rfr_bus.predict(x_test_scaled_bus))*100))
# %%
mae_rfr_bus = mean_absolute_error(y_test_b, y_pred_rfr_bus)
print('MAE score:', mae_rfr_bus)
logger.info('MAE of RFR_business Model{}'.format(mae_rfr_bus))

r2_rfr_bus = r2_score(y_test_b, y_pred_rfr_bus)
print('R2 score:', r2_rfr_bus)
logger.info('R2 score of RFR_business Model {}'.format(r2_rfr_bus))

adj_r2_rfr_bus = 1 - (1-r2_score(y_test_b, y_pred_rfr_bus)) * (len(y)-1)/(len(y)-X.shape[1]-1)
print('adjusted_R-squared:', adj_r2)
logger.info('Adjusted R2 score of RFR_business Model {}'.format(adj_r2_rfr_bus))

rfr_bus_end = time.time()
print('Random Forest Regressor Business Class time:', (rfr_bus_end - rfr_start_bus))