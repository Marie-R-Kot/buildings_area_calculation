"""
Streamlit-приложение для анализа площади застройки.

Позволяет указать адрес для анализа в поле ввода или выбрать область на карте
Отображаем снимок спутника и маску, показываем пользователю процент и площадь застройки
"""

import streamlit as st

from streamlit_folium import st_folium

from utils.geo_tools import get_coords_by_address
from utils.streamlit_utils import (
    CHAT_ICON,
    DEFAULT_ADDRESS,
    DEFAULT_CENTER,
    DEFAULT_ZOOM,
    MAP_WARNING,
    WELCOME_MESSAGE,
    draw_analyze_section,
    draw_map_widget,
    init_session_state,
    prepare_image_download,
    stream_text_generator,
)

#
# * Формируем заголовок вкладки в браузере
# * Инициализируем пустые атрибуты сессии в st.session_state по-умолчанию
#

st.set_page_config(
    layout="centered",
    page_title="Подсчет площади застройки",
    page_icon=CHAT_ICON,
)
init_session_state(st)

#
# Постоянный заголовок – название проекта и ссылка на документацию
#

col1, col2 = st.columns([0.7, 0.3], vertical_alignment="bottom", gap="large")
with col1:
    st.subheader(":material/apartment: Подсчет площади застройки")
with col2:
    st.link_button(
        "Документация",
        "https://github.com/Marie-R-Kot",
        type="secondary",
        icon=":material/docs:",
    )
st.divider()


#
# Приветственное сообщение.
# Отображаем всегда, без условий
#

# Первый раз отображаем "набираемым текстом" (write_stream), а при ререндере – отображаем
# сразу целиком, чтобы избежать повторного ожидания эффекта "набора" при каждом ререндере

if not st.session_state.get("welcome_shown", False):
    with st.chat_message("assistant", avatar=CHAT_ICON):
        st.write_stream(stream_text_generator(WELCOME_MESSAGE))
        st.session_state.welcome_shown = True
else:
    with st.chat_message("assistant", avatar=CHAT_ICON):
        st.markdown(WELCOME_MESSAGE)

#
# Предупреждение о масштабе и виджет карты
# Отображаем всегда, без условий
#

with st.chat_message("assistant", avatar=CHAT_ICON):
    st.warning(MAP_WARNING, icon=":material/warning:")

    # Контейнер-спойлер с виджетом карты и кнопкой запуска анализе
    with st.expander(
        "Выбрать область на виджете карты",
        expanded=st.session_state.map_center != DEFAULT_CENTER,
        # разворачиваем виджет, если введён адрес (центр карты не равен стандартному)
    ):
        # Виджет карты
        map_center = st.session_state.get("map_center", DEFAULT_CENTER)
        map_widget = draw_map_widget(map_center, DEFAULT_ZOOM)
        map_output = st_folium(
            map_widget,
            width="100%",
            height=500,
            returned_objects=["center", "zoom", "bounds"],
        )
        st.session_state.bounds = map_output["bounds"]

        # Кнопка начала анализа
        analyze_btn = st.button(
            "Проанализировать площадь застройки",
            type="secondary",
            use_container_width=True,
            disabled=st.session_state.analysis_started,  # disabled после начала анализа
        )

        if analyze_btn:
            # Сохраняем zoom и границы карты с виджета для последующего анализа
            st.session_state.zoom_level = map_output["zoom"]
            st.session_state.bounds = map_output["bounds"]

            st.session_state.analysis_started = True  # Переключаем режим UI на "анализ"
            st.rerun()  # Ререндер с analysis_started=True чтобы заблокировать кнопку


#
# Отрисовываем форму ввода адреса
#
# Условие # 1: st.session_state.analysis_started = True
#
# Условие # 2: st.session_state.analysis_complete = False
# после ввода адреса, поле ввода отображать уже не будем
#

if not st.session_state.analysis_started and not st.session_state.analysis_complete:
    # Проверка, что пользователь ввёл адрес в чат (он не равен стандартному значению)
    is_address_entered = st.session_state.map_center != DEFAULT_CENTER

    text = (
        f"Здесь можно ввести адрес. Например: {DEFAULT_ADDRESS}"
        if not is_address_entered
        else "Адрес корректен! Проверь на карте, что адрес тот – и нажми кнопку анализа"
    )

    if prompt := st.chat_input(
        text,
        disabled=st.session_state.analysis_started or is_address_entered,
        # Блокируем ввод, если адрес принят, или анализ уже начат
    ):
        # Сохраняем адрес из сообщения в session_state
        st.session_state.user_address = prompt

        try:
            lat, lon = get_coords_by_address(prompt)
            st.session_state.map_center = (lat, lon)
            st.session_state.zoom_level = DEFAULT_ZOOM

            # Перезапускаем UI, чтобы мы перерисовали виджет карты с введённым адресом
            st.rerun()

        except IndexError:
            # в ```text``` форматируем пример адреса как код (с возможностью копирования)
            error_msg = (
                f"**Не удалось найти такой адрес.** "
                f"Попробуй ввести другой запрос, например: ```{DEFAULT_ADDRESS}```"
            )
            st.toast(error_msg, duration="long", icon=":material/warning:")
        except Exception as e:
            error_msg = f"Ошибка при парсинге адреса: {str(e)}"
            st.toast(error_msg, duration="long", icon=":material/warning:")


#
# Отрисовываем процесс анализа - когда выбрана область в виджете / введён правильный адрес
#
# Условие: st.session_state.analysis_started = True
#
# Условие "not st.session_state.analysis_complete" отсутствует, чтобы не потерять
# результат анализа при обновлении страницы (ререндере)
#

if st.session_state.analysis_started:
    with st.chat_message("assistant", avatar=CHAT_ICON):
        draw_analyze_section(st.session_state.bounds)
        st.session_state.analysis_complete = True


#
# Флоу "Результат анализа"
#
# Условие # 1: st.session_state.analysis_complete = True
#
# Условие # 2: получены st.session_state.satellite_image и .mask_image
# если всё получено, считаем площадь застройки и рисуем финальные кнопки скачивания
#

if (
    st.session_state.analysis_complete
    and st.session_state.satellite_image
    and st.session_state.mask_image
):
    with st.chat_message("assistant", avatar=CHAT_ICON):
        st.write("**А вот и результаты анализа:**")

        # Столбцы с метриками (процент застройки + площадь в м2)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Площадь застройки на карте",
                value=f"{st.session_state.building_percentage:.1f}%",
            )
        with col2:
            st.metric(
                label="Общая площадь зданий",
                value=f"{st.session_state.building_area_m2:,.0f} м²",
            )

        st.write("А ещё можно...")

        # Кнопки скачивания
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            img_byte_arr, filename = prepare_image_download(
                st.session_state.satellite_image, "satellite",
            )
            st.download_button(
                label="Скачать снимок",
                data=img_byte_arr,
                file_name=filename,
                mime="image/png",
                use_container_width=True,
            )

        with col_btn2:
            mask_byte_arr, mask_filename = prepare_image_download(
                st.session_state.mask_image, "mask"
            )
            st.download_button(
                label="Скачать маску",
                data=mask_byte_arr,
                file_name=mask_filename,
                mime="image/png",
                use_container_width=True,
            )

        with col_btn3:
            # Обычный st.rerun() не очищает чат красиво, поэтому просто обновляем страницу
            # браузера с помощью внедрения прямого вызова JavaScript-кода: location.reload
            st.button(
                "Начать заново",
                use_container_width=True,
                on_click=lambda: st.components.v1.html(
                    "<script>window.parent.location.reload();</script>", height=0
                ),
            )
