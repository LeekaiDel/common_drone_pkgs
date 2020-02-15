#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum


class Fly_modes(Enum):
    INIT_STATE = 0          # инициализация системы
    TAKE_OFF = 1            # взлёт на заданную высоту
    GO_TO_LINE = 2          # летим по траектории
    FIND_WINDOW = 3         # нашли окно

class Mission_state(Enum):
    WAIT = 0            # ожидание
    WORK = 1            # работа планировщикка
    COMPLETE = 2        # работа завершина
    BROKE = 3           # ошибка