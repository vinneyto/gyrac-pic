# GYRAC PIC

Исследовательская 3D электростатическая Particle-in-Cell модель водородной
плазмы для GYRAC. Ядро написано на PyTorch, поддерживает CUDA, MPS и CPU,
релятивистский Boris pusher, векторизованный CIC, matrix-free PCG, подключаемые
поля TE111 и магнитного зеркала, checkpoint и необязательную визуализацию Rerun.

> Параметры цилиндра в профиле `classic_gyrac_x` — модельные размеры идеальной
> полости, подобранные около 2.4 GHz, а не измеренные размеры установки.

## Установка и быстрый запуск

```bash
pip install -e '.[test,visualization,notebook]'
pytest
jupyter lab notebooks/
```

Минимальный API:

```python
from gyrac_pic import BoxDomain, Experiment, make_smoke_config

config = make_smoke_config()
domain = BoxDomain((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), config.grid.shape)
experiment = Experiment.create(config, domain, modules=[])
experiment.initialize()
experiment.run(10)
experiment.save_checkpoint("runs/example/final.pt")
```

`.rrd` содержит только визуализацию; физическое продолжение выполняется из
`.pt` checkpoint. Физический ramp 100 μs при стандартном шаге требует примерно
125 миллионов шагов и поэтому не запускается notebook по умолчанию. Укрупнённый
демонстрационный ramp обязан быть помечен `nonphysical_scaled_ramp=True`.

## Численные ограничения

Модель не включает собственное магнитное поле плазмы, столкновения, ионизацию,
излучение, вторичную эмиссию и обратную связь плазмы с модой. Цилиндрическая PEC
граница является stair-step маской на декартовой сетке. Перед интерпретацией
результатов проверяйте сходимость по сетке, числу частиц, шагу и PCG tolerance.
