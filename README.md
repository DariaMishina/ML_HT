## ML hometask1

 * __что было сделано:__
   - EDA и предобработка признаков,
   - модели: линейная регрессия, Lasso и Ridge,
   - подобраны гиперпараметры моделей на кроссвалидации,
   - опробованы разные варианты обучающих датасетов (только вещественные признаки, вещественные+категориальные признаки),
   - Feature Engineering (разные варианты скейлера, добавление полиномиальных признаков, логарифмирование таргета),
   - написана функция для кастомной бизнес-метрики по заданному ТЗ,
   - реализован сервис на FastAPI, скриншоты его работы [тут](https://docs.google.com/document/d/1KWmdFmU354Ia2c81gR6ym8fKrNURyjkjcx5IgjMjW54/edit?usp=sharing)
* __с какими результатами и что дало наибольший буст:__
   - наилучших результатов удалось достичь, используя полиномиальные признаки, логарифмирование таргета и Ridge - r2 на test 0.8535
   - добавление категориальных признаков к вещественным также позволило улучшить скор модели
* __что сделать не вышло и почему:__
   - не удалось сделать дополнительную визуализацию (рисование красивых графиков - не моя самая сильная сторона)
