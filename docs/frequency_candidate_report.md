# Frequency candidate v1: независимая проверка PR #24

Модуль принимается как экспериментальная библиотека с отключённым runtime. Прирост качества и основание для автоматического подключения не установлены. Проверенный source: `4ffafbba6f6ea9df8fe450fa9c2cf95827c5f159`, база `143708b91f50959052dceb15ac968ef279071f02`.

Статическая проверка не нашла P0/P1. Фиксированный baseline source и весь components recipe сохраняются; counts строятся только на fit, новые значения получают 0. Существующие runtime/defaults не менялись. Локально прошли 75 тестов: 29 pure candidate, 45 fake experiment и один tiny native test двух LogisticRegression components на 30 fit / 2 predict строках. Последний подтверждает контракт wrapper и full-refit trace, а не качество.

Реальный smoke был остановлен после control arm: `censored / nested_censored_cleanup_unverified`. Launcher завершился с кодом 0 за 22,971 с, host сохранил baseline после outer/full refit. Однако пять startup trials (ExtraTrees, два CatBoost и два LightGBM) получили timeout; process inventory недоступен, cleanup потомков не доказан. Строгий censor не переименован в успех.

Frequency arm и серия из 16 arms не запускались. Сравнимых пар — 0; frequency не измерялся, не принимался и не публиковался. Настроенный memory cap не означает измеренный RSS или доказанное enforcement. Public TRAIN supplement не запускался.

Метод smoke: `936c986064d7311f436ddd1615c6fc23de66cd41ab1308f92180e39920119ee4`. Точные пять trials, исходные hashes и команда tiny native test сохранены в [frequency_candidate_native_review.json](frequency_candidate_native_review.json). Предзарегистрированный протокол: [frequency_candidate_protocol.md](frequency_candidate_protocol.md).

Рекомендация: хранить библиотеку и стенд; до подключения провести native smoke и всю sealed серию в среде с доступной process inventory. Ни тестовый имитатор, ни этот остановленный smoke не являются evidence прироста.
