## IJCAI-18 阿里妈妈搜索广告转化预测

https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.744733afsbCSLA&raceId=231647

**赛题内容**

#### **赛题内容**

本次比赛以阿里电商广告为研究对象，提供了淘宝平台的海量真实交易数据，参赛选手通过人工智能技术构建预测模型预估用户的购买意向，即给定广告点击相关的用户（user）、广告商品（ad）、检索词（query）、上下文内容（context）、商店（shop）等信息的条件下预测广告产生购买行为的概率（pCVR），形式化定义为：pCVR=P(conversion=1 | query, user, ad, context, shop)。

结合淘宝平台的业务场景和不同的流量特点，我们定义了以下两类挑战：

（1）日常的转化率预估

（2）特殊日期的转化率预估



#### 评估指标

通过logarithmic loss（记为logloss）评估模型效果（越小越好）， 公式如下：

![TB1R0pMa29TBuNjy0FcXXbeiFXa-660-122[1].png](https://work.alibaba-inc.com/aliwork_tfs/g01_alibaba-inc_com/tfscom/TB19Ad1aY9YBuNjy0FgXXcxcXXa.tfsprivate.png)



其中N表示测试集样本数量，yi表示测试集中第i个样本的真实标签，pi表示第i个样本的预估转化率。



#### 数据说明

本次比赛为参赛选手提供了5类数据（基础数据、广告商品信息、用户信息、上下文信息和店铺信息）。基础数据表提供了搜索广告最基本的信息，以及“是否交易”的标记。广告商品信息、用户信息、上下文信息和店铺信息等4类数据，提供了对转化率预估可能有帮助的辅助信息。



用于初赛的数据包含了若干天的样本。最后一天的数据用于结果评测，对选手不公布；其余日期的数据作为训练数据，提供给参赛选手；。



在上述各张数据表中，绝大部分样本包含了完整的字段数据，也有少部分样本缺乏特定字段的数据。如果一条样本的某个字段为“-1”，表示这个样本的对应字段缺乏数据。



#### **基础数据**

| 字段        | 解释                                                         |
| ----------- | ------------------------------------------------------------ |
| instance_id | 样本编号，Long                                               |
| is_trade    | 是否交易的标记位，Int类型；取值是0或者1，其中1 表示这条样本最终产生交易，0 表示没有交易 |
| item_id     | 广告商品编号，Long类型                                       |
| user_id     | 用户的编号，Long类型                                         |
| context_id  | 上下文信息的编号，Long类型                                   |
| shop_id     | 店铺的编号，Long类型                                         |

#### **广告商品信息**

| 字段                 | 解释                                                         |
| -------------------- | ------------------------------------------------------------ |
| item_id              | 广告商品编号，Long类型                                       |
| item_category_list   | 广告商品的的类目列表，String类型；从根类目（最粗略的一级类目）向叶子类目（最精细的类目）依次排列，数据拼接格式为 "category_0;category_1;category_2"，其中 category_1 是 category_0 的子类目，category_2 是 category_1 的子类目 |
| item_property_list   | 广告商品的属性列表，String类型；数据拼接格式为 "property_0;property_1;property_2"，各个属性没有从属关系 |
| item_brand_id        | 广告商品的品牌编号，Long类型                                 |
| item_city_id         | 广告商品的城市编号，Long类型                                 |
| item_price_level     | 广告商品的价格等级，Int类型；取值从0开始，数值越大表示价格越高 |
| item_sales_level     | 广告商品的销量等级，Int类型；取值从0开始，数值越大表示销量越大 |
| item_collected_level | 广告商品被收藏次数的等级，Int类型；取值从0开始，数值越大表示被收藏次数越大 |
| item_pv_level        | 广告商品被展示次数的等级，Int类型；取值从0开始，数值越大表示被展示次数越大 |

#### **用户信息**

| 字段               | 解释                                                         |
| ------------------ | ------------------------------------------------------------ |
| user_id            | 用户的编号，Long类型                                         |
| user_gender_id     | 用户的预测性别编号，Int类型；0表示女性用户，1表示男性用户，2表示家庭用户 |
| user_age_level     | 用户的预测年龄等级，Int类型；数值越大表示年龄越大            |
| user_occupation_id | 用户的预测职业编号，Int类型                                  |
| user_star_level    | 用户的星级编号，Int类型；数值越大表示用户的星级越高          |

#### **上下文信息**

| 字段                      | 解释                                                         |
| ------------------------- | ------------------------------------------------------------ |
| context_id                | 上下文信息的编号，Long类型                                   |
| context_timestamp         | 广告商品的展示时间，Long类型；取值是以秒为单位的Unix时间戳，以1天为单位对时间戳进行了偏移 |
| context_page_id           | 广告商品的展示页面编号，Int类型；取值从1开始，依次增加；在一次搜索的展示结果中第一屏的编号为1，第二屏的编号为2 |
| predict_category_property | 根据查询词预测的类目属性列表，String类型；数据拼接格式为 “category_A:property_A_1,property_A_2,property_A_3;category_B:-1;category_C:property_C_1,property_C_2” ，其中 category_A、category_B、category_C 是预测的三个类目；property_B 取值为-1，表示预测的第二个类目 category_B 没有对应的预测属性 |

#### **店铺信息**

| 字段                      | 解释                                                         |
| ------------------------- | ------------------------------------------------------------ |
| shop_id                   | 店铺的编号，Long类型                                         |
| shop_review_num_level     | 店铺的评价数量等级，Int类型；取值从0开始，数值越大表示评价数量越多 |
| shop_review_positive_rate | 店铺的好评率，Double类型；取值在0到1之间，数值越大表示好评率越高 |
| shop_star_level           | 店铺的星级编号，Int类型；取值从0开始，数值越大表示店铺的星级越高 |
| shop_score_service        | 店铺的服务态度评分，Double类型；取值在0到1之间，数值越大表示评分越高 |
| shop_score_delivery       | 店铺的物流服务评分，Double类型；取值在0到1之间，数值越大表示评分越高 |
| shop_score_description    | 店铺的描述相符评分，Double类型；取值在0到1之间，数值越大表示评分越高 |