import streamlit as st

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer

from PIL import Image

import pickle
import warnings
warnings.filterwarnings(action='ignore')

import seaborn as sns
sns.set_style('whitegrid')


st.sidebar.title("Phân loại Email với các thuật toán học máy")
st.sidebar.markdown("Email của bạn là: ")
st.sidebar.markdown("✅Email ham hay 🚫Email spam🍄")

trichrut = st.sidebar.radio(
     "Chọn phương pháp trích rút đặc trưng",
     ("Bag of words", "TF-IDF"))
#hiển thị các biểu đồ cho đánh giá chung
def plot_metrics(metrics_list,image1,image2):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if 'Độ chính xác' in metrics_list:
        st.subheader("Biểu đồ độ chính xác 7 thuật toán")
        st.image(image1, caption='Biểu đồ độ chính xác 7 thuật toán')

    if 'Thời gian đào tạo' in metrics_list:
        st.subheader("Biểu đồ thời gian đào tạo 7 thuật toán")
        st.image(image2, caption='Biểu đồ thời gian đào tạo 7 thuật toán')
#hiển thị các biểu đồ cho đánh giá từng thuật toán
def plot_metrics_one_algorithm(metrics_list,image1,image2,image3):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        st.image(image1, caption='Ma trận nhầm lẫn')

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        st.image(image2, caption='Biểu đồ đường cong ROC')

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        st.image(image3, caption='Biểu đồ đường cong Precision Recall')


def clean_str(string, reg=RegexpTokenizer(r'[a-z]+')):
    # Clean a string with RegexpTokenizer
    string = string.lower()
    tokens = reg.tokenize(string)
    return " ".join(tokens)

def stemming(text):
    stemmer = PorterStemmer()
    return ''.join([stemmer.stem(word) for word in text])



def load_models_algorithm_count():
    random_model = pickle.load(open("models/CountVectorizer/Random Forest.pkl", "rb"))
    Muti_model = pickle.load(open("models/CountVectorizer/MultinomialNB.pkl", "rb"))
    logistic_model = pickle.load(open("models/CountVectorizer/Logistic Regression.pkl", "rb"))
    knn_model = pickle.load(open("models/CountVectorizer/KNN.pkl", "rb"))
    tree_model = pickle.load(open("models/CountVectorizer/Decision Tree.pkl", "rb"))
    linearsvc_model = pickle.load(open("models/CountVectorizer/SVM (Linear).pkl", "rb"))
    svc_model = pickle.load(open("models/CountVectorizer/SVM (RBF).pkl", "rb"))
    return random_model, Muti_model, logistic_model, knn_model, tree_model, linearsvc_model, svc_model
random_model1, Muti_model1, logistic_model1, knn_model1, tree_model1, linearsvc_model1, svc_model1 = load_models_algorithm_count()


def load_models_algorithm_tfidf():
    random_model = pickle.load(open("models/TfidfVectorizer/Random Forest.pkl", "rb"))
    Muti_model = pickle.load(open("models/TfidfVectorizer/MultinomialNB.pkl", "rb"))
    logistic_model = pickle.load(open("models/TfidfVectorizer/Logistic Regression.pkl", "rb"))
    knn_model = pickle.load(open("models/TfidfVectorizer/KNN.pkl", "rb"))
    tree_model = pickle.load(open("models/TfidfVectorizer/Decision Tree.pkl", "rb"))
    linearsvc_model = pickle.load(open("models/TfidfVectorizer/SVM (Linear).pkl", "rb"))
    svc_model = pickle.load(open("models/TfidfVectorizer/SVM (RBF).pkl", "rb"))
    return random_model, Muti_model, logistic_model, knn_model, tree_model, linearsvc_model, svc_model
random_model2, Muti_model2, logistic_model2, knn_model2, tree_model2, linearsvc_model2, svc_model2 = load_models_algorithm_tfidf()
if trichrut == "Bag of words":
    classifier = st.sidebar.selectbox("Đánh giá",
                                      ("Đánh giá chung",
                                       "Từng thuật toán"))
    if classifier == 'Đánh giá chung':
        df_raw = pickle.load(open("models/df_raw.pkl","rb"))
        rowall = df_raw['Label_Number'].value_counts()
        row0 = rowall[0]
        # so luong dong 1 -spam email
        row1 = rowall[1]
        #phần trăm dữ liệu và số lượng dữ liệu
        col1, col2 = st.columns(2)
        with col1:
            st.header("Phần trăm dữ liệu")
            labels_circe = 'Spam', 'Not Spam'
            sizes = [row1, row0]
            explode = (0, 0.1) #hiện nổi bật
            fig1, ax1 = plt.subplots(1)
            ax1.pie(sizes, explode=explode, labels=labels_circe, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig1)
        with col2:
            st.header("Số lượng dữ liệu")
            fig2 = plt.figure(figsize=(8, 6))
            sns.countplot(data=df_raw, x='Category')
            st.pyplot(fig2)
        #các biểu đồ
        metrics = st.sidebar.multiselect("Chọn các biểu đồ?",
                                         ('Độ chính xác', 'Thời gian đào tạo'))
        #biểu đồ độ chính xác -thời gian đào tạo
        image1 = Image.open('img_models/CountVectorizer/plot_acc_count.png')
        image2 = Image.open('img_models/CountVectorizer/plot_time_count.png')
        plot_metrics(metrics,image1,image2)
        if st.sidebar.checkbox("Hiển thị bảng độ chính xác", True):
            st.subheader("Bảng so sánh độ chính xác 7 thuật toán")
            table_acc_count = pickle.load(open("models/table_acc_count.pkl","rb"))
            st.write(table_acc_count)
        if st.sidebar.checkbox("Hiển thị bảng các độ đo khác", True):
            st.subheader("Bảng so sánh 7 thuật toán qua các độ đo khác nhau")
            table_metrics_count = pickle.load(open("models/table_metrics_count.pkl","rb"))
            st.write(table_metrics_count)
        st.balloons()
    if classifier == 'Từng thuật toán':
        table_metrics_count = pickle.load(open("models/table_metrics_count.pkl", "rb"))
        classifier = st.selectbox("Classification Algorithms",
                                          ("Random Forest Classifier",
                                           "Multinomial Naïve Bayes",
                                           "Logistic Regression",
                                           "K Nearest Neighbor",
                                           "Decision Tree Classifier",
                                           "Support Vector Machine (Linear)",
                                           "Support Vector Machine (RBF)"))
        if classifier == 'Random Forest Classifier':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Random Forest Classifier")
            #model = random
            accuracy = table_metrics_count['Accuracy score'][0]
            precision = table_metrics_count['Precision score'][0]
            recall = table_metrics_count['Recall score'][0]
            f1score = table_metrics_count['F1 score'][0]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/CountVectorizer/Random Forest/cfm.png')
            image2 = Image.open('img_models/CountVectorizer/Random Forest/roc.png')
            image3 = Image.open('img_models/CountVectorizer/Random Forest/precision_recall.png')
            plot_metrics_one_algorithm(metrics,image1,image2,image3)
        if classifier == 'Multinomial Naïve Bayes':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Multinomial Naïve Bayes")
            #model = random
            accuracy = table_metrics_count['Accuracy score'][1]
            precision = table_metrics_count['Precision score'][1]
            recall = table_metrics_count['Recall score'][1]
            f1score = table_metrics_count['F1 score'][1]
            st.write("Accuracy ", accuracy.round(1))
            st.write("Precision: ", precision.round(1))
            st.write("Recall: ", recall.round(1))
            st.write("F1 score: ", f1score.round(1))

            image1 = Image.open('img_models/CountVectorizer/MultinomialNB/cfm.png')
            image2 = Image.open('img_models/CountVectorizer/MultinomialNB/roc.png')
            image3 = Image.open('img_models/CountVectorizer/MultinomialNB/precision_recall.png')
            plot_metrics_one_algorithm(metrics,image1,image2,image3)

        if classifier == 'Logistic Regression':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Logistic Regression")
            #model = random
            accuracy = table_metrics_count['Accuracy score'][2]
            precision = table_metrics_count['Precision score'][2]
            recall = table_metrics_count['Recall score'][2]
            f1score = table_metrics_count['F1 score'][2]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/CountVectorizer/Logistic Regression/cfm.png')
            image2 = Image.open('img_models/CountVectorizer/Logistic Regression/roc.png')
            image3 = Image.open('img_models/CountVectorizer/Logistic Regression/precision_recall.png')
            plot_metrics_one_algorithm(metrics,image1,image2,image3)
        if classifier == 'K Nearest Neighbor':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("K Nearest Neighbor")
            #model = random
            accuracy = table_metrics_count['Accuracy score'][3]
            precision = table_metrics_count['Precision score'][3]
            recall = table_metrics_count['Recall score'][3]
            f1score = table_metrics_count['F1 score'][3]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/CountVectorizer/KNN/cfm.png')
            image2 = Image.open('img_models/CountVectorizer/KNN/roc.png')
            image3 = Image.open('img_models/CountVectorizer/KNN/precision_recall.png')
            plot_metrics_one_algorithm(metrics,image1,image2,image3)
        if classifier == 'Decision Tree Classifier':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Decision Tree Classifier")
            #model = random
            accuracy = table_metrics_count['Accuracy score'][4]
            precision = table_metrics_count['Precision score'][4]
            recall = table_metrics_count['Recall score'][4]
            f1score = table_metrics_count['F1 score'][4]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/CountVectorizer/Decision Tree/cfm.png')
            image2 = Image.open('img_models/CountVectorizer/Decision Tree/roc.png')
            image3 = Image.open('img_models/CountVectorizer/Decision Tree/precision_recall.png')
            plot_metrics_one_algorithm(metrics,image1,image2,image3)
        if classifier == 'Support Vector Machine (Linear)':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Support Vector Machine (Linear)")
            #model = random
            accuracy = table_metrics_count['Accuracy score'][5]
            precision = table_metrics_count['Precision score'][5]
            recall = table_metrics_count['Recall score'][5]
            f1score = table_metrics_count['F1 score'][5]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/CountVectorizer/SVM (Linear)/cfm.png')
            image2 = Image.open('img_models/CountVectorizer/SVM (Linear)/roc.png')
            image3 = Image.open('img_models/CountVectorizer/SVM (Linear)/precision_recall.png')
            plot_metrics_one_algorithm(metrics,image1,image2,image3)
        if classifier == 'Support Vector Machine (RBF)':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Support Vector Machine (RBF)")
            #model = random
            accuracy = table_metrics_count['Accuracy score'][6]
            precision = table_metrics_count['Precision score'][6]
            recall = table_metrics_count['Recall score'][6]
            f1score = table_metrics_count['F1 score'][6]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/CountVectorizer/SVM (RBF)/cfm.png')
            image2 = Image.open('img_models/CountVectorizer/SVM (RBF)/roc.png')
            image3 = Image.open('img_models/CountVectorizer/SVM (RBF)/precision_recall.png')
            plot_metrics_one_algorithm(metrics,image1,image2,image3)
        if st.sidebar.checkbox("Hiển thị bảng dữ liệu", True):
            st.subheader("Bảng dữ liệu ban đầu")
            df_raw = pickle.load(open("models/df_raw.pkl","rb"))
            st.dataframe(df_raw)
        st.balloons()
        # giao dien side bar dự đoán

        msg = st.sidebar.text_area("Nhập nội dung email: ",height=100)
        txt1 = clean_str(msg)
        txt2 = stemming(txt1)
        data1 = [txt2]
        cv = pickle.load(open("vectorizer1.pkl","rb"))
        vect = cv.transform(data1).toarray()
        if st.sidebar.button("Predict"):
            if classifier == 'Random Forest Classifier':
                prediction1 = random_model1.predict(vect)
            elif classifier == 'Multinomial Naïve Bayes':
                prediction1 = Muti_model1.predict(vect)
            elif classifier == 'Logistic Regression':
                prediction1 = logistic_model1.predict(vect)
            elif classifier == 'K Nearest Neighbor':
                prediction1 = knn_model1.predict(vect)
            elif classifier == 'Decision Tree Classifier':
                prediction1 = tree_model1.predict(vect)
            elif classifier == 'Support Vector Machine (Linear)':
                prediction1 = linearsvc_model1.predict(vect)
            else:
                prediction1 = svc_model1.predict(vect)

            result1 = prediction1[0]
            if result1 == 1:
                st.sidebar.error("Đây là email rác")

            else:
                st.sidebar.success("Đây là email thường")

else:
    classifier = st.sidebar.selectbox("Đánh giá",
                                      ("Đánh giá chung",
                                       "Từng thuật toán"))
    if classifier == 'Đánh giá chung':
        # phần trăm dữ liệu và số lượng dữ liệu
        df_raw = pickle.load(open("models/df_raw.pkl", "rb"))
        rowall = df_raw['Label_Number'].value_counts()
        row0 = rowall[0]
        # so luong dong 1 -spam email
        row1 = rowall[1]
        # phần trăm dữ liệu và số lượng dữ liệu
        col1, col2 = st.columns(2)
        with col1:
            st.header("Phần trăm dữ liệu")
            labels_circe = 'Spam', 'Not Spam'
            sizes = [row1, row0]
            explode = (0, 0.1)  # hiện nổi bật
            fig1, ax1 = plt.subplots(1)
            ax1.pie(sizes, explode=explode, labels=labels_circe, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig1)
        with col2:
            st.header("Số lượng dữ liệu")
            fig2 = plt.figure(figsize=(8, 6))
            sns.countplot(data=df_raw, x='Category')
            st.pyplot(fig2)
        metrics = st.sidebar.multiselect("Chọn các biểu đồ?",
                                         ('Độ chính xác', 'Thời gian đào tạo'))
        # biểu đồ độ chính xác -thời gian đào tạo
        image1 = Image.open('img_models/TfidfVectorizer/plot_acc_tfidf.png')
        image2 = Image.open('img_models/TfidfVectorizer/plot_time_tfidf.png')
        plot_metrics(metrics, image1, image2)
        if st.sidebar.checkbox("Hiển thị bảng độ chính xác", True):
            st.subheader("Bảng so sánh độ chính xác 7 thuật toán")
            table_acc_tfidf = pickle.load(open("models/table_acc_tfidf.pkl","rb"))
            st.write(table_acc_tfidf)
        if st.sidebar.checkbox("Hiển thị bảng các độ đo khác", True):
            st.subheader("Bảng so sánh 7 thuật toán qua các độ đo khác nhau")
            table_metrics_tfidf = pickle.load(open("models/table_metrics_tfidf.pkl","rb"))
            st.write(table_metrics_tfidf)
        st.balloons()

    if classifier == 'Từng thuật toán':
        table_metrics_tfidf = pickle.load(open("models/table_metrics_tfidf.pkl", "rb"))
        classifier = st.selectbox("Classification Algorithms",
                                          ("Random Forest Classifier",
                                           "Multinomial Naïve Bayes",
                                           "Logistic Regression",
                                           "K Nearest Neighbor",
                                           "Decision Tree Classifier",
                                           "Support Vector Machine (Linear)",
                                           "Support Vector Machine (RBF)"))
        if classifier == 'Random Forest Classifier':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Random Forest Classifier")
            # model = random
            accuracy = table_metrics_tfidf['Accuracy score'][0]
            precision = table_metrics_tfidf['Precision score'][0]
            recall = table_metrics_tfidf['Recall score'][0]
            f1score = table_metrics_tfidf['F1 score'][0]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/TfidfVectorizer/Random Forest/cfm.png')
            image2 = Image.open('img_models/TfidfVectorizer/Random Forest/roc.png')
            image3 = Image.open('img_models/TfidfVectorizer/Random Forest/precision_recall.png')
            plot_metrics_one_algorithm(metrics, image1, image2, image3)
        if classifier == 'Multinomial Naïve Bayes':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Multinomial Naïve Bayes")
            # model = random
            accuracy = table_metrics_tfidf['Accuracy score'][1]
            precision = table_metrics_tfidf['Precision score'][1]
            recall = table_metrics_tfidf['Recall score'][1]
            f1score = table_metrics_tfidf['F1 score'][1]
            st.write("Accuracy ", accuracy.round(1))
            st.write("Precision: ", precision.round(1))
            st.write("Recall: ", recall.round(1))
            st.write("F1 score: ", f1score.round(1))

            image1 = Image.open('img_models/TfidfVectorizer/MultinomialNB/cfm.png')
            image2 = Image.open('img_models/TfidfVectorizer/MultinomialNB/roc.png')
            image3 = Image.open('img_models/TfidfVectorizer/MultinomialNB/precision_recall.png')
            plot_metrics_one_algorithm(metrics, image1, image2, image3)

        if classifier == 'Logistic Regression':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Logistic Regression")
            # model = random
            accuracy = table_metrics_tfidf['Accuracy score'][2]
            precision = table_metrics_tfidf['Precision score'][2]
            recall = table_metrics_tfidf['Recall score'][2]
            f1score = table_metrics_tfidf['F1 score'][2]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/TfidfVectorizer/Logistic Regression/cfm.png')
            image2 = Image.open('img_models/TfidfVectorizer/Logistic Regression/roc.png')
            image3 = Image.open('img_models/TfidfVectorizer/Logistic Regression/precision_recall.png')
            plot_metrics_one_algorithm(metrics, image1, image2, image3)
        if classifier == 'K Nearest Neighbor':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("K Nearest Neighbor")
            # model = random
            accuracy = table_metrics_tfidf['Accuracy score'][3]
            precision = table_metrics_tfidf['Precision score'][3]
            recall = table_metrics_tfidf['Recall score'][3]
            f1score = table_metrics_tfidf['F1 score'][3]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/TfidfVectorizer/KNN/cfm.png')
            image2 = Image.open('img_models/TfidfVectorizer/KNN/roc.png')
            image3 = Image.open('img_models/TfidfVectorizer/KNN/precision_recall.png')
            plot_metrics_one_algorithm(metrics, image1, image2, image3)
        if classifier == 'Decision Tree Classifier':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Decision Tree Classifier")
            # model = random
            accuracy = table_metrics_tfidf['Accuracy score'][4]
            precision = table_metrics_tfidf['Precision score'][4]
            recall = table_metrics_tfidf['Recall score'][4]
            f1score = table_metrics_tfidf['F1 score'][4]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/TfidfVectorizer/Decision Tree/cfm.png')
            image2 = Image.open('img_models/TfidfVectorizer/Decision Tree/roc.png')
            image3 = Image.open('img_models/TfidfVectorizer/Decision Tree/precision_recall.png')
            plot_metrics_one_algorithm(metrics, image1, image2, image3)
        if classifier == 'Support Vector Machine (Linear)':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Support Vector Machine (Linear)")
            # model = random
            accuracy = table_metrics_tfidf['Accuracy score'][5]
            precision = table_metrics_tfidf['Precision score'][5]
            recall = table_metrics_tfidf['Recall score'][5]
            f1score = table_metrics_tfidf['F1 score'][5]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/TfidfVectorizer/SVM (Linear)/cfm.png')
            image2 = Image.open('img_models/TfidfVectorizer/SVM (Linear)/roc.png')
            image3 = Image.open('img_models/TfidfVectorizer/SVM (Linear)/precision_recall.png')
            plot_metrics_one_algorithm(metrics, image1, image2, image3)
        if classifier == 'Support Vector Machine (RBF)':
            metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                             ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            st.subheader("Support Vector Machine (RBF)")
            # model = random
            accuracy = table_metrics_tfidf['Accuracy score'][6]
            precision = table_metrics_tfidf['Precision score'][6]
            recall = table_metrics_tfidf['Recall score'][6]
            f1score = table_metrics_tfidf['F1 score'][6]
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            st.write("F1 score: ", f1score.round(4))

            image1 = Image.open('img_models/TfidfVectorizer/SVM (RBF)/cfm.png')
            image2 = Image.open('img_models/TfidfVectorizer/SVM (RBF)/roc.png')
            image3 = Image.open('img_models/TfidfVectorizer/SVM (RBF)/precision_recall.png')
            plot_metrics_one_algorithm(metrics, image1, image2, image3)
        if st.sidebar.checkbox("Hiển thị bảng dữ liệu", True):
            st.subheader("Bảng dữ liệu ban đầu")
            df_raw = pickle.load(open("models/df_raw.pkl", "rb"))
            st.dataframe(df_raw)
        st.balloons()
        # giao dien side bar dự đoán

        msg = st.sidebar.text_area("Nhập nội dung email: ", height=100)
        txt1 = clean_str(msg)
        txt2 = stemming(txt1)
        data1 = [txt2]
        cv = pickle.load(open("vectorizer2.pkl", "rb"))
        vect = cv.transform(data1).toarray()
        if st.sidebar.button("Predict"):
            if classifier == 'Random Forest Classifier':
                prediction1 = random_model2.predict(vect)
            elif classifier == 'Multinomial Naïve Bayes':
                prediction1 = Muti_model2.predict(vect)
            elif classifier == 'Logistic Regression':
                prediction1 = logistic_model2.predict(vect)
            elif classifier == 'K Nearest Neighbor':
                prediction1 = knn_model2.predict(vect)
            elif classifier == 'Decision Tree Classifier':
                prediction1 = tree_model2.predict(vect)
            elif classifier == 'Support Vector Machine (Linear)':
                prediction1 = linearsvc_model2.predict(vect)
            else:
                prediction1 = svc_model2.predict(vect)

            result1 = prediction1[0]
            if result1 == 1:
                st.sidebar.error("Đây là email rác")

            else:
                st.sidebar.success("Đây là email thường")