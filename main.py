from server.api.endpoints import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # экземпляр нашего приложения
        host="127.0.0.1",  # слушать на всех интерфейсах
        port=8001,      # порт для подключения
        reload=True     # автоматическая перезагрузка при изменениях
    )