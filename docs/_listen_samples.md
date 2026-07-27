# 试听清单（主线源 poc_qwen_asr_full_result.json, 1350 条）
听法: 先听 enrollment 记住 target 声音 → 听 recognition(原始带噪) → 听切片(切的效果) → 对比 ref/qwen/vanilla


##### ★ 成功: 切对+转对 (qwen_CER=0) (候选 680 条, 取 3) #####

  uid=cmd_0  qwen_CER=0.000  sim=0.254  bucket=[0.2,0.4) 主战场  切片=3.66s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_0.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_0.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_0.wav
    ref(正确)   : 空调开到制热调到二十五度风量调到百分之三十
    qwen转写    : 空调开到制热，调到二十五度，风量调到百分之三十。
    vanilla转写 : 空调开到自热调到25度风量调到30%   (vanilla_CER=0.429)

  uid=cmd_1  qwen_CER=0.000  sim=0.294  bucket=[0.2,0.4) 主战场  切片=2.84s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_1.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_1.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_1.wav
    ref(正确)   : 灯光亮度调到百分之三十
    qwen转写    : 灯光亮度调到百分之三十。
    vanilla转写 : 灯光亮度调到30%   (vanilla_CER=0.455)

  uid=cmd_10  qwen_CER=0.000  sim=0.349  bucket=[0.2,0.4) 主战场  切片=1.74s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_10.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_10.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_10.wav
    ref(正确)   : 关闭悬浮空调
    qwen转写    : 关闭悬浮空调。
    vanilla转写 : 關閉懸浮空調   (vanilla_CER=0.667)

##### 中等 (qwen_CER 0.2-0.6) (候选 189 条, 取 3) #####

  uid=cmd_151  qwen_CER=0.200  sim=0.524  bucket=>=0.4 接近解决  切片=1.34s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_151.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_151.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_151.wav
    ref(正确)   : 我要轻干洗
    qwen转写    : 我要清干洗。
    vanilla转写 : 我要清乾洗   (vanilla_CER=0.400)

  uid=cmd_170  qwen_CER=0.200  sim=0.193  bucket=<0.2 死区  切片=1.68s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_170.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_170.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_170.wav
    ref(正确)   : 一键净呼吸
    qwen转写    : 一键静呼吸。
    vanilla转写 : 一見進呼吸   (vanilla_CER=0.400)

  uid=cmd_190  qwen_CER=0.200  sim=0.342  bucket=[0.2,0.4) 主战场  切片=0.98s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_190.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_190.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_190.wav
    ref(正确)   : 我要吃饭咯
    qwen转写    : 我要吃饭喽。
    vanilla转写 : 我要吃飯囉   (vanilla_CER=0.400)

##### ✗ 失败: 切错或转崩 (qwen_CER>0.8) (候选 344 条, 取 4) #####

  uid=cmd_2491  qwen_CER=0.800  sim=0.194  bucket=<0.2 死区  切片=1.63s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2491.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2491.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2491.wav
    ref(正确)   : 把空调关闭
    qwen转写    : 把空调关闭掉，可以吗？
    vanilla转写 : 把副作公立给了   (vanilla_CER=1.200)

  uid=cmd_2663  qwen_CER=0.800  sim=0.374  bucket=[0.2,0.4) 主战场  切片=1.74s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2663.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2663.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2663.wav
    ref(正确)   : 产妇哪些食物需要忌口
    qwen转写    : 有哪些手机壳？
    vanilla转写 : 这下手机   (vanilla_CER=1.000)

  uid=cmd_2665  qwen_CER=0.800  sim=0.313  bucket=[0.2,0.4) 主战场  切片=1.82s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2665.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2665.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2665.wav
    ref(正确)   : 风速调最小
    qwen转写    : 大小。
    vanilla转写 : 那是中大小   (vanilla_CER=0.800)

  uid=cmd_2702  qwen_CER=0.800  sim=0.351  bucket=[0.2,0.4) 主战场  切片=0.62s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2702.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2702.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2702.wav
    ref(正确)   : 播放温妮女巫魔法绘本
    qwen转写    : 播放。
    vanilla转写 : 我放   (vanilla_CER=0.900)

##### 极短切片 <0.6s (共 12 条, 取最短 5) — 看是不是切废了 #####

  uid=cmd_2002  qwen_CER=1.000  sim=0.318  bucket=[0.2,0.4) 主战场  切片=0.30s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2002.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2002.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2002.wav
    ref(正确)   : 开屏幕
    qwen转写    : 险。
    vanilla转写 : 谢谢   (vanilla_CER=1.000)

  uid=cmd_2291  qwen_CER=1.000  sim=0.112  bucket=<0.2 死区  切片=0.34s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2291.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2291.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2291.wav
    ref(正确)   : 给我讲三百六十五读书
    qwen转写    : 哎。
    vanilla转写 : 好   (vanilla_CER=1.000)

  uid=cmd_2771  qwen_CER=1.000  sim=0.025  bucket=<0.2 死区  切片=0.34s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2771.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2771.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2771.wav
    ref(正确)   : 开启智清洁
    qwen转写    : 哎。
    vanilla转写 : OK   (vanilla_CER=1.000)

  uid=cmd_2890  qwen_CER=1.000  sim=0.425  bucket=>=0.4 接近解决  切片=0.36s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2890.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2890.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2890.wav
    ref(正确)   : 吃什么有利于脂肪肝
    qwen转写    : 你车。
    vanilla转写 : 的墙   (vanilla_CER=1.000)

  uid=cmd_324  qwen_CER=1.000  sim=0.422  bucket=>=0.4 接近解决  切片=0.37s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_324.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_324.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_324.wav
    ref(正确)   : 就这样子把风速减小
    qwen转写    : 嗯。
    vanilla转写 : 好   (vanilla_CER=1.000)

##### 反例: sim≥0.4 却转崩 (共 43 条, 取 4) — 证'sim高≠切对' #####

  uid=cmd_2942  qwen_CER=1.500  sim=0.605  bucket=>=0.4 接近解决  切片=1.70s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2942.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2942.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2942.wav
    ref(正确)   : 开防直吹
    qwen转写    : 五十一分完赛。
    vanilla转写 : 51分完赛   (vanilla_CER=1.250)

  uid=cmd_2251  qwen_CER=1.000  sim=0.604  bucket=>=0.4 接近解决  切片=1.24s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2251.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2251.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2251.wav
    ref(正确)   : 把温度调到三十度
    qwen转写    : 行政区域内的。
    vanilla转写 : 行政区域内的   (vanilla_CER=1.000)

  uid=cmd_2637  qwen_CER=1.125  sim=0.585  bucket=>=0.4 接近解决  切片=2.09s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2637.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2637.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2637.wav
    ref(正确)   : 哺乳期要少吃什么
    qwen转写    : 就已经进行过一轮交。
    vanilla转写 : 就已经进行过一轮教育   (vanilla_CER=1.250)

  uid=cmd_2659  qwen_CER=1.200  sim=0.584  bucket=>=0.4 接近解决  切片=1.70s
    [听target原声] enrollment : E:/midea_target_asr/datasetA/pos/kws_2659.wav
    [听原始带噪] recognition : E:/midea_target_asr/datasetA/pos/cmd_2659.wav
    [听切的效果] 切片        : E:/target_slices_full/cmd_2659.wav
    ref(正确)   : 空调十八度
    qwen转写    : 顶尖选手参赛。
    vanilla转写 : 内定间纯手参赛   (vanilla_CER=1.400)