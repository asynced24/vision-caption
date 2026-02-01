from app import build_demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue()
    demo.launch(share=True)
