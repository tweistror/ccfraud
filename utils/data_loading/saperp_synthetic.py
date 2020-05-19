import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler

# TODO: Put in preprocessing folder with extra class

def get_data_saperp(dataset_string, path, fraud_only):
    path = path['one']

    if dataset_string == "saperp-ek":
        x_ben, x_fraud = load_EK(path, fraud_only)
    else:
        x_ben, x_fraud = load_VK(path, fraud_only)

    return x_ben, x_fraud


def load_EK(path, fraud_only):
    df = load_data_EK_gen(path)
    df = prepare_data_EK_gen(df)

    if fraud_only:
        x_ben = df.loc[df['Label'].apply(lambda x: not x.endswith('fraud'))]
        x_fraud = df.loc[df['Label'].apply(lambda x: x.endswith('fraud'))]
    else:
        x_ben = df.loc[df['Label'] == 'regular']
        x_fraud = df.loc[df['Label'] != 'regular']

    x_ben.drop(['Label'], axis=1, inplace=True)
    x_fraud.drop(['Label'], axis=1, inplace=True)

    return x_ben, x_fraud


def load_VK(path, fraud_only):
    df = load_data_VK_gen(path)
    df = prepare_data_VK_gen(df)

    if fraud_only:
        x_ben = df.loc[df['Label'].apply(lambda x: not x.endswith('fraud'))]
        x_fraud = df.loc[df['Label'].apply(lambda x: x.endswith('fraud'))]
    else:
        x_ben = df.loc[df['Label'] == 'regular']
        x_fraud = df.loc[df['Label'] != 'regular']

    x_ben.drop(['Label'], axis=1, inplace=True)
    x_fraud.drop(['Label'], axis=1, inplace=True)

    return x_ben, x_fraud


def load_data_EK_gen(path):
    # read relevant features from csv df_EK = pd.read_excel(io='./data/generated/generated_purchasing_data.xlsx',
    # sheet_name='EKPO_Data', usecols=['Angelegt von', 'Lieferant', 'Material', 'Werk', 'Bestellmenge',
    # 'Bestellnettopreis', 'Angelegt am', 'Uhrzeit', 'Label'])
    df_EK = pd.read_csv(filepath_or_buffer=f'{path}/generated_purchasing_data.csv', sep=';', encoding='utf-8',
                        usecols=['Einkaufsbeleg', 'Angelegt von', 'Lieferant', 'Position', 'Material', 'Werk',
                                 'Bestellmenge', 'Bestellnettopreis', 'Angelegt am', 'Uhrzeit'],
                        converters={'Angelegt am': lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'),
                                    'Uhrzeit': lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time()})

    df_EK_invoice = pd.read_csv(filepath_or_buffer=f'{path}/generated_purchasing_invoices.csv', sep=';',
                                encoding='utf-8',
                                usecols=['Referenz', 'ReferenzPosition', 'RechnungsBelegnummer', 'RechnungsPosition'])

    df_EK_payments = pd.read_csv(filepath_or_buffer=f'{path}/generated_payments.csv', sep=';',
                                 encoding='utf-8',
                                 usecols=['Referenz', 'ReferenzPosition', 'Kontonummer', 'Soll/Haben-Kennz.', 'Label'])
    df_EK_payments = df_EK_payments[df_EK_payments['Soll/Haben-Kennz.'] == 'S']

    df_EK = df_EK.merge(df_EK_invoice, left_on=['Einkaufsbeleg', 'Position'], right_on=['Referenz', 'ReferenzPosition'])
    df_EK = df_EK.merge(df_EK_payments, left_on=['RechnungsBelegnummer', 'RechnungsPosition'],
                        right_on=['Referenz', 'ReferenzPosition'])

    # df_EK = df_EK.drop(columns=['Einkaufsbeleg','Referenz', 'RechnungsBelegnummer', 'Belegnummer',
    # 'Soll/Haben-Kennz.'])

    df_EK['Kontonummer'] = df_EK['Kontonummer'].astype(str)
    df_EK = df_EK[['Lieferant', 'Material', 'Werk', 'Angelegt von', 'Kontonummer', 'Bestellmenge', 'Bestellnettopreis',
                   'Angelegt am', 'Uhrzeit', 'Label']]

    return df_EK


def load_data_VK_gen(path):
    df_VK = pd.read_csv(filepath_or_buffer=f'{path}/generated_sales_data.csv', sep=';', encoding='utf-8',
                        usecols=['Verkaufsbeleg', 'Angelegt von', 'Auftraggeber', 'Position', 'Material', 'Werk',
                                 'Auftragsmenge', 'Nettopreis', 'Angelegt am', 'Uhrzeit'],
                        converters={'Angelegt am': lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'),
                                    'Uhrzeit': lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time()})
    df_VK['Auftraggeber'] = df_VK['Auftraggeber'].astype(str)

    df_VK_deliveries = pd.read_csv(filepath_or_buffer=f'{path}/generated_sales_deliveries.csv', sep=';',
                                   encoding='utf-8',
                                   usecols=['Referenz', 'ReferenzPosition', 'Übergabeort', 'Label'])

    df_VK = df_VK.merge(df_VK_deliveries, left_on=['Verkaufsbeleg', 'Position'],
                        right_on=['Referenz', 'ReferenzPosition'])

    return df_VK


def prepare_data_EK_gen(df_EK, use_numeric=True, log_scale=False):
    df_label = df_EK['Label']

    # convert categorical features into dummy variables (one-hot encoding)
    df_EK_categ_trans = pd.get_dummies(df_EK[['Lieferant', 'Material', 'Werk', 'Angelegt von', 'Kontonummer']])

    df_EK_numeric_norm = df_EK[['Bestellmenge', 'Bestellnettopreis']]
    df_EK_datetime = df_EK[['Angelegt am', 'Uhrzeit']]
    df_EK_core_time = pd.DataFrame(index=df_EK.index, columns=['in_core_time'], dtype=int)

    for index, row in df_EK_datetime.iterrows():
        on_weekday = 0
        in_core_time = 0
        if row['Angelegt am'].isoweekday() in range(1, 6):
            on_weekday = 1
        if on_weekday == 1 \
                and row['Uhrzeit'] > datetime.time(8, 00) \
                and row['Uhrzeit'] < datetime.time(18, 00):
            in_core_time = 1
        df_EK_core_time.loc[index] = in_core_time

    df_EK = df_EK_categ_trans.join(df_EK_core_time)

    if use_numeric:
        # apply normalization
        df_EK_numeric_norm = pd.DataFrame(MinMaxScaler().fit_transform(df_EK_numeric_norm),
                                          index=df_EK_numeric_norm.index, columns=df_EK_numeric_norm.columns)

        # union preprocessed numerical features and one-hot categorical features
        df_EK = df_EK.join(df_EK_numeric_norm)

    df_EK = df_EK.join(df_label)

    return df_EK


def prepare_data_VK_gen(df_VK, use_numeric=True, log_scale=False):
    df_label = df_VK['Label']

    # convert categorical features into dummy variables (one-hot encoding)
    df_VK_categ_trans = pd.get_dummies(df_VK[['Auftraggeber', 'Material', 'Angelegt von', 'Übergabeort']])

    df_VK_numeric_norm = df_VK[['Auftragsmenge', 'Nettopreis']]
    df_VK_datetime = df_VK[['Angelegt am', 'Uhrzeit']]
    df_VK_core_time = pd.DataFrame(index=df_VK.index, columns=['in_core_time'], dtype=int)

    for index, row in df_VK_datetime.iterrows():
        on_weekday = 0
        in_core_time = 0
        if row['Angelegt am'].isoweekday() in range(1, 6):
            on_weekday = 1
        if on_weekday == 1 \
                and datetime.time(8, 00) < row['Uhrzeit'] < datetime.time(18, 00):
            in_core_time = 1
        df_VK_core_time.loc[index] = in_core_time

    # union preprocessed numerical features and one-hot categorical features
    df_VK = df_VK_categ_trans.join(df_VK_core_time)

    if use_numeric:
        # apply normalization
        df_VK_numeric_norm = pd.DataFrame(MinMaxScaler().fit_transform(df_VK_numeric_norm),
                                          index=df_VK_numeric_norm.index, columns=df_VK_numeric_norm.columns)

        df_VK = df_VK.join(df_VK_numeric_norm)

    df_VK = df_VK.join(df_label)

    return df_VK
