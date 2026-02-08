"""Вспомогательные функции и параметры конфигурации для streamlit"""

import io

from datetime import datetime

import folium
import streamlit as st

from folium import Element

from utils.geo_tools import get_bounds, get_image
from utils.model_interaction import predict_image

# Конфигурация работы Streamlit

DEFAULT_ZOOM = 17
DEFAULT_CENTER = [41.9082, -87.7227]
DEFAULT_ADDRESS = "1500 N Hamlin Ave, Чикаго"
MODEL_FILE = "best_model.pth"

# Инициализация атрибутов session_state


def init_session_state(st):
    """Инициализация состояний приложения – выставляем значения атрибутов по умолчанию"""

    # Отслеживаем старт анализа, чтобы блокировать формы ввода и кнопку под виджетом
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    # Отслеживаем конец анализа, чтобы посчитать и вывести в UI результат
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False

    # Храним различные параметры виджета, результаты анализа и изображения
    if "map_center" not in st.session_state:
        st.session_state.map_center = DEFAULT_CENTER

    if "zoom_level" not in st.session_state:
        st.session_state.zoom_level = DEFAULT_ZOOM

    if "final_image" not in st.session_state:
        st.session_state.final_image = None

    if "mask_image" not in st.session_state:
        st.session_state.mask_image = None

    if "building_percentage" not in st.session_state:
        st.session_state.building_percentage = 0.0

    if "building_area_m2" not in st.session_state:
        st.session_state.building_area_m2 = 0.0

    if "bounds" not in st.session_state:
        st.session_state.bounds = None

    if "real_resolution" not in st.session_state:
        st.session_state.real_resolution = 0.0


# Красиво выводим приветственный текст :)


def stream_text_generator(text, delay=0.01):
    """
    Генератор для st.write_stream - постепенный вывод текста.
    """
    import time

    for char in text:
        yield char
        time.sleep(delay)


def prepare_image_download(image, prefix="image"):
    """Подготовка изображения для скачивания"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    return img_byte_arr, filename


# Текст сообщений, предупреждений и иконок, чтобы не дублировать их в коде

WELCOME_MESSAGE = """Привет! Давай посчитаем площадь застройки. Для этого тебе нужно: 
1. Либо выбрать нужную область на карте, развернув спойлер
2. Или же просто ввести адрес внизу и отправить сообщение :)"""

MAP_WARNING = """Важно! Чем меньше масштаб карты, тем дольше будут скачиваться 
изображения карт. Для больших областей это может занять несколько минут."""

CHAT_ICON = ":material/robot:"


# Функции-модули streamlit, которые мы вызываем в коде несколько раз


def draw_map_widget(center: tuple = DEFAULT_CENTER, zoom: int = DEFAULT_ZOOM):
    """Создание виджета с картой"""
    map = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        zoom_control=True,
        scrollWheelZoom=False,
        touchZoom=False,
        doubleClickZoom=False,
        dragging=True,
    )

    # Убираем лишние надписи в виджете
    css_hide_attribution = """
    <style>.leaflet-control-attribution { display: none !important; }</style>
    """
    map.get_root().header.add_child(Element(css_hide_attribution))

    return map


def draw_analyze_section(bounds: dict = None):
    """
    Анализ области и вывод результатов анализа в интерфейс.

    Вынесено в отдельную функцию, чтобы объединить флоу двух разных сценариев:
    * нажатие кнопки под виджетом карты
    * отправка адреса в чате
    """

    # Спойлер с загрузкой спутникового изображения
    with st.status("Скачивание спутниковых снимков...", expanded=False) as status:
        if not st.session_state.get("satellite_image"):
            corrent_bounds = get_bounds(bounds)
            satellite_image, real_resolution = get_image(corrent_bounds)

            if satellite_image is None:
                st.error("Не удалось загрузить изображение")

            # Сохраняем изображение и разрешение в session_state
            st.session_state.satellite_image = satellite_image
            st.session_state.real_resolution = real_resolution

        # Спутниковое изображение (рисуем в спойлере)
        with st.expander("Спутниковый снимок участка", expanded=False):
            st.image(
                st.session_state.satellite_image,
                caption="Спутниковый снимок максимально возможного разрешения, "
                "склеенный из тайлов",
                width="content",
            )

        # Меняем статус на "complete" после завершения анализа
        status.update(
            label="Скачивание и склейка тайлов завершены", state="complete", expanded=True
        )

    # Спойлер с анализом снимка и выводом результатов
    with st.status("Анализ снимка с помощью модели...", expanded=False) as status:
        if not st.session_state.get("mask_image"):
            mask_image, area, percent = predict_image(
                MODEL_FILE,
                st.session_state.satellite_image,
                st.session_state.real_resolution,
            )

            # Сохраняем маску и результат анализа в состоянии
            st.session_state.mask_image = mask_image
            st.session_state.building_percentage = percent
            st.session_state.building_area_m2 = area

        # Изображение маски из модели (рисуем в спойлере)
        with st.expander("Маска зданий", expanded=False):
            st.image(
                st.session_state.mask_image,
                caption="Маска застройки – белым цветом отмечены здания",
                width="content",
            )

        # Меняем статус на "complete" после завершения анализа
        status.update(
            label="Анализ и создание маски завершены", state="complete", expanded=True
        )
