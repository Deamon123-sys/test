import streamlit as st
import pandas as pd

# Định nghĩa hàm hiển thị hộp thoại (popup)
@st.dialog("Popup Dialog")
def open_dialog(message, func, arg):
    st.write(message)  # Hiển thị thông báo
    if st.button("Confirm"):
        func(arg)  # Gọi hàm `func` với đối số `arg` khi nhấn "Confirm"
        st.session_state.open_dialog = None  # Đặt trạng thái `open_dialog` về None
        st.rerun()  # Chạy lại ứng dụng

    if st.button("Cancel"):
        st.session_state.open_dialog = None  # Đặt trạng thái `open_dialog` về None nếu hủy
        st.rerun()  # Chạy lại ứng dụng


# Định nghĩa hộp thoại hiển thị danh sách các collection
@st.dialog("Collection List", width="large")
def list_collection(session_state, load_func, delete_func):
    # Kiểm tra nếu client đã tồn tại trong session_state
    if "client" in session_state and session_state.client:
        collections = session_state.client.list_collections()  # Lấy danh sách collection

        # Chuẩn bị dữ liệu để hiển thị trong DataFrame
        collection_data = [
            {
                "Collection Name": collection.name,  # Tên của collection
                "Metadata": str(collection.metadata)  # Metadata của collection
            }
            for collection in collections
        ]

        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(collection_data, columns=["Collection Name", "Metadata", "Action"])

        # Tạo các cột tiêu đề cho bảng
        head_col1, head_col2, head_col3 = st.columns([2, 2, 2])
        with head_col1:
            st.write("**Collection Name**")  # Hiển thị tiêu đề "Collection Name"
        with head_col2:
            st.write("**Metadata**")  # Hiển thị tiêu đề "Metadata"
        with head_col3:
            st.write("**Action**")  # Hiển thị tiêu đề "Action"
        st.markdown("---")  # Tạo đường kẻ ngang để phân cách

        # Duyệt từng hàng trong DataFrame và hiển thị
        for index, row in df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                st.write(row["Collection Name"])  # Hiển thị tên của collection
                if st.session_state.random_collection_name == row["Collection Name"]:
                    st.markdown("""
                        <span style="color: red;">This collection is currently in use.</span>
                    """, unsafe_allow_html=True)  # Hiển thị thông báo nếu collection đang được sử dụng
            with col2:
                st.write(row["Metadata"])  # Hiển thị metadata của collection
            with col3:
                # Nút "Load" để tải collection
                if st.button("Load", key=f"load_{index}"):
                    load_func(row["Collection Name"])  # Gọi hàm `load_func` với tên collection
                    st.rerun()  # Chạy lại ứng dụng
            with col4:
                # Nút "Delete" để xóa collection
                if st.button("Delete", key=f"delete_{index}"):
                    delete_func(row["Collection Name"])  # Gọi hàm `delete_func` để xóa collection
                    st.rerun()  # Chạy lại ứng dụng
            st.markdown("---")  # Tạo đường kẻ ngang giữa các collection
