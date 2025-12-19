# 自動リリース手順

このプロジェクトは、**GitHub Actionsによる完全自動リリース**を採用しています。

## 🚀 自動リリースの流れ

1. **developブランチからmainへのマージ（推奨）** または **mainブランチへの直接プッシュ** → 自動的に以下が実行されます：
   - バージョンを自動インクリメント（デフォルト: パッチバージョン）
   - リリースブランチ（`release/v0.2.0`）を作成
   - `pyproject.toml`のバージョンを更新
   - パッケージをビルド
   - PyPIに公開
   - GitHub Releaseを作成
   - mainブランチにマージ

## 📋 使用方法

### 方法1: developブランチからのマージ（推奨）

```bash
# developブランチで開発
git checkout develop
git add .
git commit -m "Add new feature"
git push origin develop

# Pull Requestを作成してmainにマージ
# → マージ時に自動的にリリースされます
```

### 方法2: mainブランチへの直接プッシュ

```bash
git add .
git commit -m "Add new feature"
git push origin main
```

→ 自動的にパッチバージョン（例: 0.1.0 → 0.1.1）がインクリメントされてリリースされます。

### 方法3: 手動でバージョンを指定

GitHub ActionsのUIから手動実行：

1. GitHubリポジトリの「Actions」タブを開く
2. 「Auto Release」ワークフローを選択
3. 「Run workflow」をクリック
4. バージョンを指定（例: `0.2.0`）またはインクリメントタイプを選択
5. 「Run workflow」を実行

### 方法4: コミットメッセージで制御（将来実装予定）

コミットメッセージに特定のキーワードを含めることで、バージョンタイプを指定：

- `[major]` → メジャーバージョンをインクリメント（0.1.0 → 1.0.0）
- `[minor]` → マイナーバージョンをインクリメント（0.1.0 → 0.2.0）
- `[patch]` → パッチバージョンをインクリメント（0.1.0 → 0.1.1）

## 🔍 リリースの確認

### GitHub Actions

1. リポジトリの「Actions」タブでワークフローの実行状況を確認
2. 各ステップのログを確認

### PyPI

- プロジェクトページ: https://pypi.org/project/galr/
- 新しいバージョンが公開されているか確認

### GitHub Releases

- リリースページ: https://github.com/yut0takagi/galr/releases
- 自動生成されたリリースノートを確認

## ⚙️ セットアップ

### GitHub Secretsの設定

1. GitHubリポジトリの Settings > Secrets and variables > Actions
2. `PYPI_API_TOKEN` を追加
   - 値: PyPIのAPIトークン（`pypi-...`形式）
   - 取得方法: https://pypi.org/manage/account/token/

## 📦 バージョン管理

### セマンティックバージョニング

- **MAJOR** (1.0.0): 後方互換性のない変更
- **MINOR** (0.2.0): 後方互換性のある新機能追加
- **PATCH** (0.1.1): バグ修正

### バージョンの自動インクリメント

- デフォルト: パッチバージョン（0.1.0 → 0.1.1）
- 手動実行時: バージョンタイプを選択可能

## 🛠️ トラブルシューティング

### リリースがスキップされる

**原因**: 同じバージョンが既に存在する

**対処**: 
- 手動実行で新しいバージョンを指定
- または、次のコミットで自動的にインクリメントされる

### PyPIへの公開が失敗する

**原因**: 
- `PYPI_API_TOKEN`が設定されていない
- トークンが無効

**対処**:
1. GitHub Secretsで`PYPI_API_TOKEN`を確認
2. PyPIで新しいトークンを生成
3. Secretsを更新

### リリースブランチが残る

通常は自動的に削除されますが、エラーが発生した場合は手動で削除：

```bash
git push origin --delete release/v0.2.0
```

## 📝 注意事項

1. **developブランチからのmainへのマージで自動リリースが実行されます**
   - 通常の開発フロー: `develop` → Pull Request → `main`（マージ時に自動リリース）
   - mainブランチへの直接プッシュもリリースをトリガーします
   - 開発中は`develop`ブランチや`feature`ブランチを使用することを推奨

2. **バージョンは自動的にインクリメントされます**
   - 意図しないリリースを避けるため、mainブランチへのマージは慎重に
   - Pull Requestをマージする前に内容を確認

3. **同じバージョンは再リリースできません**
   - 既存のバージョンが存在する場合、リリースはスキップされます

4. **Pull Requestがマージされなかった場合はリリースされません**
   - マージされなかったPRではリリースは実行されません
