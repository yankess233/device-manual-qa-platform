from minio import Minio

# 初始化客户端
client = Minio(
    "127.0.0.1:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # 是否使用https
)

if not client.bucket_exists("knowledge"):
    client.make_bucket("knowledge")

# 上传文件
client.fput_object(
    bucket_name="knowledge",
    object_name="images/p1.jpg",  # Object名称含路径
    file_path=r"C:\Users\yan\Pictures\Screenshots\1.png",  # 上传的本地文件路径
    content_type="image/jpeg"  # MIME类型
)

# 上传后访问地址
url = "http://127.0.0.1:9000/knowledge/images/p1.jpg"
print(url)