import streamlit as st
#streamlit run c:/Users/rrmkh/Documents/ML/TITANIC_SURVIVAL_PREDICTION/app.py



class web_app:
    st.title(":blue[Titanic survival predictor]", text_alignment='center')
    st.divider()
    with st.container(border=True):
        st.subheader('On April 15, 1912, the **RMS [Titanic](https://en.wikipedia.org/wiki/Titanic) sank after striking an iceberg** — claiming 1,502 of 2,224 lives. This project uses passenger [data](https://www.kaggle.com/datasets/yasserh/titanic-dataset) to understand who survived and build a model that can predict survival from'
        ' key features like class, age, sex, and fare')
    st.divider()
        
    with st.container(border=True, gap='small'):    
        col1, col2, col3 = st.columns(3)
    #(Supported native colors: blue, green, orange, red, violet, gray, and rainbow.)
      
     
        with col1:
            st.metric(label=":blue[Total Passengers]", value=":violet[891]")
        with col2:
            st.metric(label=":blue[Features used]", value=":violet[12]")
        with col3:
            st.metric(label="Model", value="Logistic Reg.")
    st.divider()
    with st.container(border=True):
        st.caption(":rainbow['🔑 Strongest finding: Sex was the most predictive feature at −54% correlation, followed by passenger class at −34%. A woman in 1st class had over a 90% chance of survival; a man in 3rd class had under 15%.]" , text_alignment="center")
    st.divider(width=900)
    st.caption(":blue[For more background details, check out the official [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset).]",text_alignment='center')
    st.divider(width=900)
    st.caption(":rainbow[Get the Github repo at [Github](https://github.com/agastyashah96-ops/TITANIC_SURVIVAL_PREDICTOR)")
    st.header(':violet[Alternatively you can also download the data set through this code]')
    st.code("""import kagglehub

# Download latest version
    path = kagglehub.dataset_download("yasserh/titanic-dataset")

    print("Path to dataset files:", path)""" , language='python')
    #..\.venv\Scripts\streamlit run main.py
