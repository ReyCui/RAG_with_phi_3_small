# RAG_with_phi_3_small
用一杯奶茶的时间，手搓一个简单的RAG  
报告日期：2024年8月18日  
项目负责人：Rey Cui   

#### 一、项目概述：

本项目使用RAG（Retrieval Augemented Generation，检索增强生成）技术对 microsoft/phi-3-small-128k-instruct 模型进行微调，目标是开发一款智能的人力资源（HR）问答机器人，该机器人能够利用大型语言模型针对私有数据进行高效准确的问题回答。请注意，目前仅支持txt格式的文档格式，后续会增加其他格式的支持。
![image](https://github.com/user-attachments/assets/3aeb89f7-ba08-4d48-b863-a8f43aa1cb38)


#### 二、技术方案与实施步骤

2.1 模型选择  

在本项目中，我们选择了 microsoft/phi-3-small-128k-instruct 这个预训练模型作为基础模型来进行微调。以下是选择该模型的理由和RAG模型的优势分析：

2.1.1 模型特点:
- 大小适中: microsoft/phi-3-small-128k-instruct 是一个相对较小的模型，这使得它在资源消耗方面更为友好，同时仍然具备足够的能力来处理复杂任务。
- 指令遵循: 该模型经过了特定的指令微调，这意味着它能够更好地理解和执行给定的指令。
- 灵活性: 它可以被用于多种自然语言处理任务，包括文本生成、问题回答等。

2.1.2 RAG模型的优势:
- 结合检索与生成: RAG模型能够从外部知识源中检索相关信息，并将其与模型自身的知识结合起来生成答案。这种结合方式能够显著提高答案的准确性。
- 上下文相关性: 通过检索最相关的文档片段，RAG模型能够提供更加精确和具体的回答。
- 扩展知识: RAG模型允许模型访问最新和最相关的信息，这对于快速变化的领域尤其重要

2.2、数据的构建： 

2.2.1 数据收集: 
- 本项目使用的是txt格式，文档详细介绍了对GraphRAG的框架解读。 

2.2.2 数据预处理:
- 读取文档，并清洗数据，去除无关的内容及空行等。

2.2.3 向量化处理:
- 使用 CharacterTextSplitter 类对原始文档进行分割
- 使用嵌入模型 NV-Embed-QA 将每个文本块转换为向量表示。
- 使用 FAISS 建立一个向量数据库，用于存储这些向量表示，方便后续的检索操作。


3、**功能整合**（进阶版RAG必填）：  介绍进阶的语音功能、Agent功能、多模态等功能的整合策略与实现方法。

#### 三、实施步骤：

3.1 环境搭建：

3.1.1 安装Anaconda或Miniconda:  
- 如果您尚未安装Anaconda或Miniconda，可以从其官方网站下载并安装。
- 如果您已经有了Python环境，并且不需要使用Anaconda或Miniconda的特性，您可以跳过此步骤。
3.1.2 创建Python 3.8虚拟环境:  
- 打开Anaconda Prompt。
- 运行以下命令创建一个新的Python 3.8环境：
```bash
conda create --name ai_endpoint python=3.8
```
3.1.3 激活环境:
- 使用以下命令激活新环境：
```bash
conda activate ai_endpoint 
```
3.1.4 安装必要的包:
- 安装所需的Python包：
```python
# 创建python 3.8虚拟环境
conda create --name ai_endpoint python=3.8
# 进入虚拟环境
conda activate ai_endpoint
# 安装nvidia_ai_endpoint工具
pip install langchain-nvidia-ai-endpoints
# 安装Jupyter Lab
pip install jupyterlab
# 安装langchain_core
pip install langchain_core
# 安装langchain
pip install langchain
pip install -U langchain-community
# 安装matplotlib
pip install matplotlib
# 安装Numpy
pip install numpy
# 安装faiss, 这里安装CPU版本
pip install faiss-cpu==1.7.4
# 安装OPENAI库
pip install openai
```
3.1.5 测试环境是否搭建成功：
- 打开jupyter-lab
```bash
jupyter-lab
```
- 在jupyter-lab中打开一个notebook，输入以下代码，测试是否安装成功：
```python
# 测试faiss是否安装成功
import faiss
print(faiss.__version__)
```

3.2. 代码实现： 
![image](https://github.com/user-attachments/assets/f321b9e8-0726-4fe0-ad0c-e0665fba6716)


![image](https://github.com/user-attachments/assets/562af456-23a6-4f80-9faf-2e19828f44d4)




#### 四、项目成果与展示：

原始模型的回答示例：
![image](https://github.com/user-attachments/assets/c0c3e40d-79d4-49a2-b714-4ec34e7db73d)

RAG模型的回答示例：
![image](https://github.com/user-attachments/assets/34ab13de-feb9-47c6-a6f9-529901991ec5)

通过上述对比，可以明显看出RAG模型的回答更加准确和精确。RAG模型利用了文档中的具体信息来构建答案，而原始模型则可能基于一般的知识或模糊的关联来进行回答。这种差异对于需要高度准确信息的企业环境尤为重要。

#### 五、问题与解决方案：

![image](https://github.com/user-attachments/assets/d953c2f0-ebed-4585-af33-e46333a74cf9)


5.1 问题描述:    
在尝试使用 FAISS.from_texts 时遇到了上述错误，并且发现问题出在 CharacterTextSplitter 的配置上。
- 文档的段落不能太长：如果文档中的段落超过了 chunk_size 的大小，那么 CharacterTextSplitter 无法正确处理。
- separator 参数只能是一个：您尝试使用 separator=[" ", "\n"]，但发现这会导致问题。

5.2 解决方案:
- 确保文档段落不超过 chunk_size：
  - 可以考虑预先处理文档，确保每个段落的长度不超过 chunk_size。
  - 如果文档中包含很长的段落，可以手动将其分割成更小的部分。
- 使用正确的 separator 参数：
  - 根据您的描述，您需要使用单个分隔符，例如 separator="\n"。
  - 这意味着 CharacterTextSplitter 会根据换行符来分割文本。

5.3 经验总结：  
选择合适的 chunk_size 对于确保文档被正确分割非常重要，同时也需要考虑到下游任务的需求和模型的限制。以下是关于 chunk_size 选择的一些优化建议：
 
当 chunk_size 设置得过大时，可能导致以下问题：
- 对于较短的问题，匹配的内容可能不够长，导致过多的内容被合并到同一个 chunk 中，从而使其中的一些细节信息丢失。
- 较大的 chunk 可能会增加下游语言模型（LLM）的负担，特别是在使用较小的模型时，处理长文本的效果通常不如处理较短文本的效果好。

建议根据嵌入模型的最大 token 数量来选择 chunk_size 的大小：
- 如果嵌入模型的最大 token 数量为 512，则建议尝试 chunk_size 在 200 至 400 个字符之间。
- 如果嵌入模型的最大 token 数量为 1024，则建议尝试 chunk_size 在 500 至 1000 个字符之间。
 
选择合适的 chunk_size：
- 根据您的文档类型和下游任务的需求选择合适的 chunk_size。
- 如果文档较短或下游任务关注细节信息，可以尝试较小的 chunk_size。
- 如果文档较长或需要保持更多的上下文信息，可以尝试较大的 chunk_size。

平衡性能和效果：
- 较小的 chunk_size 可以帮助保留更多的细节信息，但可能会增加处理开销。
- 较大的 chunk_size 可以减少处理开销，但可能会影响模型的处理效果，特别是对于较小的模型。

逐步调整：
- 从建议的范围内开始尝试 chunk_size，然后逐步调整以找到最佳的平衡点。
观察分割效果和下游任务的表现，以确定最合适的 chunk_size。


#### 六、项目总结与展望：

6.1 项目评估  
- 成功点:
  - 成功实现了基于RAG技术的智能HR问答机器人的基本功能。
  - 能够有效地处理和回答关于人力资源相关的常见问题。
  - 利用了microsoft/phi-3-small-128k-instruct模型，该模型具有良好的指令遵循能力和语言生成能力。
  - 实现了文档的预处理和向量化，为后续的检索提供了坚实的基础。
- 存在的不足:
  - 目前仅支持txt格式的文档，限制了系统的应用场景和数据来源。
  - 在处理复杂或长篇幅文档时，可能存在向量化效率和准确性的挑战。
  - 用户输入的prompt有时存在歧义，需要进一步优化输入解析机制。
6.2 未来方向  
- 增加文档格式支持:
  - 增加对HTML和PDF等格式的支持，以扩大系统的适用范围。
  - 可以考虑使用现有的库或工具来解析这些文档，如使用pdfminer.six来解析PDF文档，使用BeautifulSoup来解析HTML文档。
- 提升用户交互体验:
  - 优化用户输入的解析逻辑，确保系统能够更准确地理解用户的意图。
  - 引入更高级的自然语言处理技术，如情感分析、实体识别等，以提供更个性化的服务。
- 性能优化:
  - 提升向量化和检索的效率，特别是在处理大量文档时。
  - 探索分布式处理和并行计算技术，以提高系统的响应速度和处理能力。
- 安全性和隐私保护:
  - 加强对私有数据的安全性和隐私保护措施。
  - 实现数据加密存储和传输，确保敏感信息的安全。
通过这些未来的改进方向和发展规划，可以进一步完善智能HR问答机器人的功能，提高其在实际场景中的应用价值和用户体验。

#### 七、特别感谢

特别感谢 NVIDIA AI 训练营的各位老师和同学，为项目提供了 invaluable 的帮助。[nvidia.cn/training/online/](https://www.nvidia.cn/training/online/)

#### 八、附件与参考资料
 
环境安装参考：https://blog.csdn.net/kunhe0512/article/details/140910139  
GraphRAG框架解读：https://www.cnblogs.com/fanzhidongyzby/p/18252630/graphrag
