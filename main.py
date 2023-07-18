import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# 加载数据
@st.cache_data # 作用是缓存数据，首次加载时将数据缓存起来，后续直接使用
def load_data():
    # 打开文件，并跳过首行
    df_agg = pd.read_csv("./youtu_data/Aggregated_Metrics_By_Video.csv").iloc[1:, :]
    # 重命名列名
    df_agg.columns = ['Video', 'Video title', 'Video publish time', 'Comments added', 'Shares', 'Dislikes', 'Likes', 'Subscribers lost',
                  'Subscribers gained', 'RPM(USD)', 'CPM(USD)', 'Average % viewed', 'Average view duration', 'Views',
                  'Watch time(hours)', 'Subscribers', 'Your estimated revenue (USD)', 'Impressions', 'Impressions ctr']

    # 修改发布时间为 datetime格式 format='mixed' 列中的格式不一致，加上mixed这个使匹配多种格式
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format='mixed')
    # 将平均观看时间的格式改成 %H:%M:%S 形式
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    # 平均每个视频观看秒数
    df_agg['Average view_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute * 60 + x.hour * 3600)
    # 参与度
    df_agg['Engagement_ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Dislikes'] + df_agg['Likes'])/ df_agg.Views
    # 观看 / 新增订阅 比
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    # 根据 Video publish time 排序
    df_agg.sort_values('Video publish time', ascending=False, inplace=True)
    df_agg_sub = pd.read_csv("./youtu_data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv")
    df_conmments = pd.read_csv("./youtu_data/All_Comments_Final.csv")
    df_time = pd.read_csv("./youtu_data/Video_Performance_Over_Time.csv")
    df_time['Date'] = pd.to_datetime(df_time['Date'], format='mixed')

    return df_agg, df_agg_sub, df_conmments, df_time


def negative_color(v, prop = ''):
    try:
        if v < 0 :
            return prop
    except:
        pass
def postive_color(v, prop = ''):
    try:
        if v >= 0 :
            return prop
    except:
        pass


def audience_simple(country):
    """Show top countries"""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'



df_agg, df_agg_sub, df_conmments, df_time = load_data()
# 聚合数据
df_agg_copy = df_agg.copy() # 赋值一份数据操作，以防弄乱原始数据

metric_date_12mo = df_agg_copy['Video publish time'].max() - pd.DateOffset(months=12)   # 获取距最新视频发布时间一年的视频日期
# 获取 12 个月以内数据每列的中位数,仅限数值列    以12个月为一个周期
df_agg_median = df_agg_copy[df_agg_copy['Video publish time'] >= metric_date_12mo].median(numeric_only=True)
# 获取数值型的列   返回一个boolean列表，数值列为 True,反之为 False   上述求的中位数就是这些数值列列
numeric_cols = np.array((df_agg_copy.dtypes == 'float64')|(df_agg_copy.dtypes == 'int64'))
# 将中位数表示基准，将数值列减掉他们的中位数，结果正或负表示上升或者下降，在除以中位数表示上升或下降的比率
df_agg_copy.iloc[:,numeric_cols] = (df_agg_copy.iloc[:,numeric_cols] - df_agg_median).div(df_agg_median)

# 合并每日数据和发布数据来获得 数据增量 delta
# 通过 External Video ID 将 Video publish time 关联到 df_time 上
df_time_diff = pd.merge(df_time, df_agg.loc[:, ['Video', 'Video publish time']], left_on='External Video ID', right_on='Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

# get last 12 months of data rather than all data 获取最近12个月的数据
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months=12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

# get daily view data (first 30), median & percentiles
views_days = pd.pivot_table(df_time_diff_yr,index= 'days_published',values ='Views', aggfunc = [np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
views_days = views_days[views_days['days_published'].between(0,30)]
views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']]
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()


st.write('## Kee Jee Youtube vedio Analysis')
#bulid dashboard
add_sidebar = st.sidebar.selectbox('Aggregated or Individual Video Analysis', ('Aggregated Metrics', 'Individual Video'))

# 视频汇总分析页
if add_sidebar == 'Aggregated Metrics':
    st.write("### 近6个月对比近12个月数据展示——以中位数为基准")
    df_agg_metrics = df_agg[['Video publish time','Views', 'Comments added', 'Shares', 'Likes', 'Subscribers', 'RPM(USD)', 'CPM(USD)', 'Average view_sec', 'Engagement_ratio', 'Views / sub gained']]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median(numeric_only=True)
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median(numeric_only=True)
    # 创建5个列
    col1, col2, col3, col4, col5 = st.columns(5)
    colums = [col1, col2, col3, col4, col5]

    count = 0
    # 遍历每个平均值
    for i in metric_medians6mo.index:
        # 遍历每一列, with 语句内的内容就会放到这一列里
        with colums[count]:
            # 获取近六个月对比近12个月各指标是上升了还是下降了
            delta = (metric_medians6mo[i] - metric_medians12mo[i]) / metric_medians12mo[i]
            # 放入 st.
            st.metric(label=i, value=round(metric_medians6mo[i], 1), delta="{:.2%}".format(delta))
            count += 1
            if(count >= 5):
                count = 0
    st.write('### 近12个月数据汇总')
    # 实现下面的总报表，选取部分数据展示
    df_agg_copy['Publish Date'] = df_agg_copy['Video publish time'].apply(lambda x: x.date())
    df_agg_new = df_agg_copy[['Video title', 'Publish Date', 'Comments added', 'Shares', 'Likes', 'Subscribers', 'RPM(USD)', 'CPM(USD)', 'Average view_sec', 'Engagement_ratio', 'Views / sub gained']]
    # 格式化数值数据
    df_agg_numeric_lst = df_agg_new.median(numeric_only=True).index.tolist() # 获得相关数值列的index
    st.write(df_agg_numeric_lst)
    df_to_pct = {}
    for i in df_agg_numeric_lst:
        df_to_pct[i] = '{:.1%}'.format  # 将每一列都赋一个格式化函数
    st.write(df_to_pct)
    # 展示数据并设置数值的颜色样式以及数值格式
    st.dataframe(df_agg_new.style.applymap(negative_color, prop = "color: red;").applymap(postive_color, prop = "color: green;").format(df_to_pct))

# 单个视频分析页
if add_sidebar == 'Individual Video':
    st.write('单个视频分析')
    # 返回 选择的 options 名，也就是视频名
    select = st.selectbox(label='选择视频', options=df_agg_copy['Video title'])
    st.write(select)
    # 获得视频信息，点赞数、订阅数 等等
    agg_filtered = df_agg[df_agg['Video title'] == select]
    st.write(agg_filtered)
    # 获得视频订阅者信息，所属国家、观看次数等等
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == select]
    st.write(agg_sub_filtered)
    # 将国家代码设置成 USA，India ,Other 三大类，以查看观众来源
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(lambda x: audience_simple(x))
    # 按 Is Subscribed 排序
    agg_sub_filtered.sort_values('Is Subscribed',inplace=True)

    # 创建观众来源图表
    st.write('观众来源')
    fig = px.bar(agg_sub_filtered, x='Views', y='Is Subscribed', color='Country', orientation='h')
    st.plotly_chart(fig)

    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0, 30)]
    first_30 = first_30.sort_values('days_published')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                              mode='lines',
                              name='20th percentile', line=dict(color='purple', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                              mode='lines',
                              name='50th percentile', line=dict(color='black', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                              mode='lines',
                              name='80th percentile', line=dict(color='royalblue', dash='dash')))
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                              mode='lines',
                              name='Current Video', line=dict(color='firebrick', width=8)))

    fig2.update_layout(title='View comparison first 30 days',
                       xaxis_title='Days Since Published',
                       yaxis_title='Cumulative views')

    st.plotly_chart(fig2)