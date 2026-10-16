from src.data import ZERO_AS_MISSING, build_preprocessor, load_data, split_data


def test_no_zeros_left_after_cleaning():
    X_train, X_test, _, _ = split_data(load_data())
    pre = build_preprocessor().fit(X_train)

    for X in (X_train, X_test):
        cleaned = pre.transform(X)
        assert (cleaned[ZERO_AS_MISSING] == 0).sum().sum() == 0


def test_pregnancies_zeros_kept():
    X_train, _, _, _ = split_data(load_data())
    cleaned = build_preprocessor().fit_transform(X_train)
    assert (cleaned["Pregnancies"] == 0).sum() == (X_train["Pregnancies"] == 0).sum()


def test_split_is_stratified():
    _, _, y_train, y_test = split_data(load_data())
    assert abs(y_train.mean() - y_test.mean()) < 0.02
