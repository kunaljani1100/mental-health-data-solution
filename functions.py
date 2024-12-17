import columnNames as c


def fillMissingData(dataset):
    """
    This function is used to deal with all the missing data in the dataset.
    :param dataset: The dataset that can either be train.csv or test.csv.
    :return: The modified dataset.
    """
    dataset[c.ACADEMIC_PRESSURE] = dataset[c.ACADEMIC_PRESSURE].fillna(0)
    dataset[c.WORK_PRESSURE] = dataset[c.WORK_PRESSURE].fillna(0)
    dataset[c.PRESSURE] = dataset[c.ACADEMIC_PRESSURE] + dataset[c.WORK_PRESSURE]
    dataset[c.PROFESSION] = dataset[c.PROFESSION].fillna('None')
    dataset[c.STUDY_SATISFACTION] = dataset[c.STUDY_SATISFACTION].fillna(0)
    dataset[c.JOB_SATISFACTION] = dataset[c.JOB_SATISFACTION].fillna(0)
    dataset[c.SATISFACTION] = dataset[c.STUDY_SATISFACTION] + dataset[
        c.JOB_SATISFACTION]
    dataset[c.CGPA] = dataset[c.CGPA].fillna(10)
    dataset[c.SATISFACTION] = 0.1 * dataset[c.SATISFACTION] * dataset[c.CGPA]
    dataset[c.DIETARY_HABITS] = dataset[c.DIETARY_HABITS].fillna('None')
    dataset[c.DEGREE] = dataset[c.DEGREE].fillna('None')
    dataset[c.FINANCIAL_STRESS] = dataset[c.FINANCIAL_STRESS].fillna('None')

    return dataset


def scaleRequiredColumns(dataset):
    """
    This function is used to scale the columns to ensure values are within a fixed range.
    :param dataset: The dataset that can either be train.csv or test.csv.
    :return: The modified dataset.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    dataset[c.AGE] = scaler.fit_transform(dataset[[c.AGE]])
    dataset[c.WORK_STUDY_HOURS] = scaler.fit_transform(dataset[[c.WORK_STUDY_HOURS]])
    dataset[c.PRESSURE] = scaler.fit_transform(dataset[[c.PRESSURE]])
    dataset[c.SATISFACTION] = scaler.fit_transform(dataset[[c.SATISFACTION]])
    dataset[c.FINANCIAL_STRESS] = dataset[c.FINANCIAL_STRESS].replace('None', 0)
    dataset[c.FINANCIAL_STRESS] = dataset[c.FINANCIAL_STRESS].astype(int)
    dataset[c.FINANCIAL_STRESS] = scaler.fit_transform(dataset[[c.FINANCIAL_STRESS]])
    return dataset


def ordinallyEncode(encoder, dataset):
    """
    This function is used to ordinally encode string columns to ensure values are within a fixed range.
    :param encoder: The encoder object used for encoding.
    :param dataset: The dataset that can either be train.csv or test.csv.
    :return: The modified dataset.
    """
    dataset[c.GENDER] = encoder.fit_transform(dataset[[c.GENDER]])
    dataset[c.CITY] = encoder.fit_transform(dataset[[c.CITY]])
    dataset[c.WORKING_PROFESSIONAL_OR_STUDENT] = encoder.fit_transform(
        dataset[[c.WORKING_PROFESSIONAL_OR_STUDENT]])
    dataset[c.PROFESSION] = encoder.fit_transform(dataset[[c.PROFESSION]])
    dataset[c.SLEEP_DURATION] = encoder.fit_transform(dataset[[c.SLEEP_DURATION]])
    dataset[c.DIETARY_HABITS] = encoder.fit_transform(dataset[[c.DIETARY_HABITS]])
    dataset[c.DEGREE] = encoder.fit_transform(dataset[[c.DEGREE]])
    dataset[c.SUICIDAL_THOUGHTS] = encoder.fit_transform(dataset[[c.SUICIDAL_THOUGHTS]])
    dataset[c.FAMILY_HISTORY_OF_MENTAL_ILLNESS] = encoder.fit_transform(
        dataset[[c.FAMILY_HISTORY_OF_MENTAL_ILLNESS]])
    return dataset
