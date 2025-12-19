# PyPI公開ガイド

このドキュメントでは、GALRパッケージをPyPIに公開する手順を説明します。

## 前提条件

1. **PyPIアカウントの作成**
   - [PyPI](https://pypi.org/account/register/)でアカウントを作成
   - [TestPyPI](https://test.pypi.org/account/register/)でもアカウントを作成（テスト用）

2. **必要なツールのインストール**
   ```bash
   pip install build twine
   ```

## 公開手順

### 1. バージョンの更新

`pyproject.toml`の`version`フィールドを更新します：

```toml
version = "0.1.0"  # 新しいバージョンに更新
```

### 2. パッケージのビルド

```bash
python -m build
```

これにより`dist/`ディレクトリに以下のファイルが生成されます：
- `galr-0.1.0.tar.gz` (ソース配布)
- `galr-0.1.0-py3-none-any.whl` (wheel配布)

### 3. ビルドの確認

```bash
# 生成されたファイルを確認
ls -lh dist/

# パッケージの内容を確認
twine check dist/*
```

### 4. TestPyPIでテスト（推奨）

初回公開時や大きな変更がある場合は、まずTestPyPIでテストします：

```bash
# TestPyPIにアップロード
twine upload --repository testpypi dist/*

# TestPyPIからインストールしてテスト
pip install --index-url https://test.pypi.org/simple/ galr
```

### 5. PyPIへのアップロード

TestPyPIでのテストが成功したら、本番のPyPIにアップロードします：

```bash
twine upload dist/*
```

認証情報を求められた場合：
- **Username**: `__token__`
- **Password**: PyPIのAPIトークン（[Account settings](https://pypi.org/manage/account/)で生成）

### 6. 公開の確認

アップロード後、以下のURLでパッケージが公開されているか確認します：

```
https://pypi.org/project/galr/
```

## トラブルシューティング

### パッケージ名が既に使用されている場合

PyPIではパッケージ名は一意である必要があります。`galr`が既に使用されている場合は、`pyproject.toml`の`name`を変更する必要があります。

### バージョンが既に存在する場合

同じバージョン番号は再アップロードできません。バージョンを更新してから再ビルド・アップロードしてください。

### 認証エラー

APIトークンが正しく設定されているか確認してください。環境変数で設定することもできます：

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## セキュリティのベストプラクティス

1. **APIトークンの管理**
   - APIトークンは環境変数や設定ファイルで管理
   - トークンをGitにコミットしない（`.gitignore`に追加済み）

2. **2要素認証（2FA）**
   - PyPIアカウントで2FAを有効化することを推奨

3. **署名付き配布（オプション）**
   - GPG署名を使用してパッケージの信頼性を向上させることも可能

## バージョン管理

セマンティックバージョニング（SemVer）に従うことを推奨します：

- **MAJOR**: 後方互換性のない変更
- **MINOR**: 後方互換性のある新機能追加
- **PATCH**: バグ修正

例：`0.1.0` → `0.1.1` (パッチ) → `0.2.0` (マイナー) → `1.0.0` (メジャー)

