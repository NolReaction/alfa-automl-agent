# Состояние работы над качеством поиска

База: main `3908292`. Ветка: `work/night-quality-v1`. Изолированные candidate/baseline worktrees созданы, исходный checkout и отправленный ZIP не изменены.

Зафиксирован `docs/night_quality_protocol.md` до первых новых обучений. Три независимые работы: pure candidate policy, pure cost scheduler, paired experiment runner. Root интегрирует только AutoML, локальный Research v3 и additive CLI option. Владение второго ИИ соблюдается.

Эксперименты пока не запускались. Внешних API calls и platform submissions: 0. Следующий этап: закончить реализацию, выполнить targeted checks, заморозить development manifest и провести конечную парную серию.
