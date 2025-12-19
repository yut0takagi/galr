# PyPIアップロード手順

## 前提条件

1. **PyPIアカウントの作成**
   - [PyPI](https://pypi.org/account/register/)でアカウントを作成
   - [TestPyPI](https://test.pypi.org/account/register/)でもアカウントを作成（テスト用）

2. **APIトークンの生成**
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - トークン名を入力して「Add token」をクリック
   - 生成されたトークン（`pypi-...`）をコピー（一度しか表示されません）

## アップロード方法

### 方法1: 環境変数を使用（推奨）

```bash
# TestPyPIにアップロード
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python3 -m twine upload --repository testpypi dist/*

# 本番PyPIにアップロード
python3 -m twine upload dist/*
```

### 方法2: コマンドラインで直接指定

```bash
# TestPyPIにアップロード
python3 -m twine upload --repository testpypi \
  --username __token__ \
  --password pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  dist/*

# 本番PyPIにアップロード
python3 -m twine upload \
  --username __token__ \
  --password pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  dist/*
```

### 方法3: .pypircファイルを使用

ホームディレクトリに `~/.pypirc` ファイルを作成：

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

その後、以下のコマンドでアップロード：

```bash
# TestPyPI
python3 -m twine upload --repository testpypi dist/*

# 本番PyPI
python3 -m twine upload dist/*
```

**注意**: `.pypirc`ファイルには機密情報が含まれるため、Gitにコミットしないでください。

## 推奨手順

1. **まずTestPyPIでテスト**
   ```bash
   python3 -m twine upload --repository testpypi dist/*
   ```

2. **TestPyPIからインストールして動作確認**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ galr
   python3 -c "from galr import GALRRegressor; print('OK')"
   ```

3. **問題なければ本番PyPIにアップロード**
   ```bash
   python3 -m twine upload dist/*
   ```

## トラブルシューティング

### パッケージ名が既に使用されている場合

エラーメッセージ: `HTTPError: 400 Client Error: File already exists`

- 同じバージョン番号は再アップロードできません
- `pyproject.toml`の`version`を更新して再ビルドしてください

### 認証エラー

エラーメッセージ: `HTTPError: 403 Client Error: Invalid or non-existent authentication information`

- APIトークンが正しいか確認
- TestPyPIとPyPIで別々のトークンが必要です
- トークンのスコープ（プロジェクト全体 vs 特定プロジェクト）を確認

