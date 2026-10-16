from src.data import ZERO_AS_MISSING, build_preprocessor, load_data, split_data


def test_no_zeros_left_after_cleaning():
    X_train, X_test, _, _ = split_data(load_data())
    pre = build_preprocessor().fit(X_train)

    for X in (X_train, X_test):
        cleaned = pre.transform(X)
        assert (cleaned[ZERO_AS_MISSING] == 0).sum().sum() == 0
        assert cleaned.isna().sum().sum() == 0


def test_real_zeros_are_kept():
    # 0 pregnancies is a valid value and must not be "fixed"
    X_train, _, _, _ = split_data(load_data())
    cleaned = build_preprocessor().fit_transform(X_train)
    assert (cleaned["Pregnancies"] == 0).sum() == (X_train["Pregnancies"] == 0).sum()


def test_split_is_stratified():
    df = load_data()
    _, _, y_train, y_test = split_data(df)
    assert len(y_train) + len(y_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
