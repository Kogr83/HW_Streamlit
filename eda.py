# загружаю бибиотеки
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# считываю файл
def create_dataset(
        path='datasets'):
    '''dataset creation'''

    # задам датасеты
    D_agreement = pd.read_csv(filepath_or_buffer=path + '/D_target' + '.csv')
    D_clients = pd.read_csv(filepath_or_buffer=path + '/D_clients' + '.csv')
    D_salary = pd.read_csv(filepath_or_buffer=path + '/D_salary' + '.csv')

    D_loan = pd.read_csv(filepath_or_buffer=path + '/D_loan' + '.csv')
    D_close_loan = pd.read_csv(filepath_or_buffer=path + '/D_close_loan' + '.csv')

    D_work = pd.read_csv(filepath_or_buffer=path + '/D_work' + '.csv')
    D_pens = pd.read_csv(filepath_or_buffer=path + '/D_pens' + '.csv')
    D_job = pd.read_csv(filepath_or_buffer=path + '/D_job' + '.csv')
    D_last_credit = pd.read_csv(filepath_or_buffer=path + '/D_last_credit' + '.csv')

    # объединю в один датафрейм D_agreement (по 'ID_CLIENT') и D_clients (по 'ID')
    new = pd.merge(D_agreement, D_clients, how='inner', left_on='ID_CLIENT', right_on='ID', indicator=True)

    # объединю в один датафрейм new и D_salary - оба по 'ID_CLIENT'
    new = pd.merge(new, D_salary, how='inner')

    # оставлю в new только нужные признаки (ДО слияния с temp)
    new = new[['ID_CLIENT', 'TARGET', 'AGE', 'GENDER',
               'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME']].copy(
        deep=True)

    # объединю в один датафрейм D_close_loan и D_loan - оба по 'ID_LOAN'
    temp = pd.merge(D_close_loan, D_loan)

    # сделаю группировку по ID_CLIENT, не забуду посчитать правильные метрики ('ID_LOAN': 'count', 'CLOSED_FL': 'sum')
    temp = temp.groupby(by=['ID_CLIENT']).agg({'ID_LOAN': 'count', 'CLOSED_FL': 'sum'}).reset_index()

    # переназову признаки, как указано в задании
    temp = temp.rename(columns={'ID_LOAN': 'LOAN_NUM_TOTAL', 'CLOSED_FL': 'LOAN_NUM_CLOSED'})

    # объединю в один датафрейм new и temp - оба по 'ID_CLIENT'
    new = pd.merge(new, temp, how='inner')

    # удалю дубликаты и выровняю индексы
    new = new.drop_duplicates().reset_index(drop=True)
    df = new.drop(['ID_CLIENT'], axis=1)

    # добавлю долю погашенных кредитов
    df['LOAN_CLOSED_SHARE'] = df['LOAN_NUM_CLOSED'] / df['LOAN_NUM_TOTAL']

    return df


# сохраняю файл
def save_dataset(path = 'datasets'):
    '''saves dataset'''
    df = create_dataset()
    df.to_csv(path_or_buf = path + '/df.csv', index = False)
    return df


# открываю файл
def open_dataset(path = "datasets/df.csv"):
    """ reads df from given path """
    df = pd.read_csv(path)
    return df

# основная статистика по количественным данным
def stat_quant():
    '''main statistics on quantitative data'''
    # перекодирую признаки
    df[['GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']] = df[
        ['GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']].astype('category')
    return df.describe()

# основная статистика по категориальным данным
def stat_cat():
    '''main statistics on categorical data'''
    # перекодирую признаки
    df[['GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']] = df[
        ['GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']].astype('category')
    return df.describe(include='category')

# отрисовка распределения основных признаков
def graphs_quant():

    '''Income distribution'''
    plt.figure(figsize=(10, 4))
    plt.hist(df['PERSONAL_INCOME'], bins=100, histtype='step', color='green')
    plt.xlabel('Уровень дохода')
    plt.ylabel('Кол-во клиентов')
    plt.title('Распределение дохода (признак №1)');

    '''Age distribution'''
    plt.figure(figsize=(10, 4))
    plt.hist(df['AGE'], bins=50, histtype='step', color='red')
    plt.xlabel('Возраст клиента')
    plt.ylabel('Кол-во клиентов')
    plt.title('Распределение клиентов по возрасту (признак №2)');

    '''Closed loans share distribution'''
    plt.figure(figsize=(10, 4))
    plt.hist(df['LOAN_CLOSED_SHARE'], bins=10, histtype='bar', color='black')
    plt.xlabel('Доля погашенных кредитов')
    plt.ylabel('Кол-во клиентов')
    plt.title('Распределение доли погашенных кредитов (признак №3)');

# матрица корреляции
def correlation():
    '''Corraltion matrix on qualnitative data'''
    # обозначу категориальные признаки
    df[['GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']] = df[
        ['GENDER', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL']].astype('category')
    # график
    plt.figure(figsize=(10, 4))
    sns.heatmap(data=df.corr(), annot=True, fmt='.2f', cmap='BrBG');

# отрисовка двухмерных зависимостей: таргет vs. признак
def graphs_causality():
    '''2 types of graphs (boxplot and histogram) on 3 dependencies:
        target vs. age
        target vs. log of income
        target vs. share of closed loans (only histogram)
        target vs. number of dependants (only histogram)
        '''
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.33)

    # 1. target vs. age
    sns.boxplot(data=df, y='AGE', x='TARGET', palette='Set1', ax=axes[0, 0])
    axes[0, 0].set_title('1. Зависимость возраста от таргета (BOXPLOT)', fontsize=11)
    axes[0, 0].set_xlabel('Таргет', fontsize=10)
    axes[0, 0].set_ylabel('Возраст', fontsize=10)

    sns.histplot(data=df, x='AGE', hue='TARGET', stat='percent', element='step', common_norm=False, ax=axes[0, 1])
    axes[0, 1].set_title('2. Зависимость возраста от таргета (ГИСТОГРАММА)', fontsize=11)
    axes[0, 1].set_xlabel('Возраст', fontsize=10)
    axes[0, 1].set_ylabel('Доля в %', fontsize=10)

    # 2. target vs. log of income
    sns.boxplot(data=df, y=np.log(df['PERSONAL_INCOME']), x='TARGET', palette='Set1', ax=axes[1, 0])
    axes[1, 0].set_title('3. Логарифм дохода от таргета (BOXPLOT)', fontsize=11)
    axes[1, 0].set_xlabel('Таргет', fontsize=10)
    axes[1, 0].set_ylabel('Логарифм дохода', fontsize=10)

    sns.histplot(data=df, x=np.log(df['PERSONAL_INCOME']), hue='TARGET', stat='percent', element='step',
                 common_norm=False, ax=axes[1, 1]);
    axes[1, 1].set_title('4. Логарифм дохода от таргета (ГИСТОГРАММА)', fontsize=11)
    axes[1, 1].set_xlabel('Логарифм дохода', fontsize=10)
    axes[1, 1].set_ylabel('Доля в %', fontsize=10)
    axes[1, 1].set_xlim(7.5, 11.5)

    # 3. target vs. share of closed loans
    sns.histplot(data=df, x='LOAN_CLOSED_SHARE', hue='TARGET', stat='percent', element='step',
                 common_norm=False, ax=axes[2, 0])
    axes[2, 0].set_title('5. Доля погашенных кредитов от таргета', fontsize=11)
    axes[2, 0].set_xlabel('Доля погашенных кредитов', fontsize=10)
    axes[2, 0].set_ylabel('Доля в таргете, %', fontsize=10)

    # 4. target vs. number of dependants
    sns.histplot(data=df, x='DEPENDANTS', hue='TARGET', stat='percent', element='step', binwidth=0.5,
                 common_norm=False, ax=axes[2, 1])
    axes[2, 1].set_title('6. Зависимость числа Dependants и таргета', fontsize=11)
    axes[2, 1].set_xlabel('Число Dependants', fontsize=10)
    axes[2, 1].set_ylabel('Доля в таргете, %', fontsize=10);

# проверка на миссинги
def mis():
    '''missing analysis'''
    print(df.isna().sum())


def cluster_analysis():
    '''KMeans cluster anaysis with standard scaler
    Returns dataset with new feature containing cluster number
    '''
    # задам скалер
    scaler = StandardScaler()

    # обучу scaler и стандартизую данные
    scaler.fit(df.drop('TARGET', axis=1))
    scaled_data = scaler.transform(df.drop('TARGET', axis=1))

    # переведу данные из numpy в pandas
    df_scaled = pd.DataFrame(scaled_data, columns=df.drop('TARGET', axis=1).columns)

    # проведу кластерный анализ
    no_of_clusters = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    temp = {}

    # проведу кластерный анализ несколько раз
    for n_clusters in no_of_clusters:
        cluster = KMeans(n_clusters=n_clusters, random_state=123)
        cluster.fit(df_scaled)
        cluster_labels = cluster.predict(df_scaled)

        # Silhouette -> качество кластеризации
        silhouette_avg = silhouette_score(df_scaled, cluster_labels)
        temp[n_clusters] = silhouette_avg

    # найду лучшее кластерное решение
    segments = list(temp.keys())
    scores = list(temp.values())
    best_segment = segments[scores.index(max(scores))]
    print(
        f'Изучено {len(no_of_clusters)} кластерных решений, лучшее решение: \n Число кластеров = {best_segment}\n Silhouette ={max(scores): .3f}')

    # сохраню модель-победитель
    kmeans_best = KMeans(n_clusters=best_segment).fit(df_scaled)
    # запишу принадлежность к кластеру в оригинальный датасет df
    df['clusters'] = kmeans_best.predict(df_scaled)
    return df


def graph_clusters():
    '''Depicts connection between clusters and target'''

    # зависимость между сегментом и таргетом
    graph = sns.histplot(data=df, x='clusters', hue='TARGET', stat='percent', element='step',
                         common_norm=False, binwidth=0.45)
    plt.title('Зависимость между сегментами и таргетом')
    plt.xlabel('Номер сегмента')
    plt.ylabel('Доля сегмента, %');


def profile_clusters():
    '''Makes segments' profile'''

    # профилирование кластеров
    profile = df.groupby(by='clusters')[['AGE', 'CHILD_TOTAL', 'PERSONAL_INCOME', 'LOAN_NUM_TOTAL',
                                         'LOAN_CLOSED_SHARE']].agg('mean').round(2)
    return profile


