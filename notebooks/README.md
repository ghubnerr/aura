## Making the best use of Colab's runtime and our repo's Source Code

So, there's two ways you can approach still developing notebooks in Google Colab while being able to access your repo's functionalities.

### 1. Running Colab hosted on your local runtime

You will want to do this when:

- You trust you don't need a GPU as good as what Google Colab can give you
- You won't need that much RAM (you're assuming your computer can handle it)
- You need the repo's source code

<b>How to set it up:</b>

1. Install these on your environment `aura`:

```bash
conda install -c conda-forge jupyter_http_over_ws
```

```bash
jupyter server extension enable --py jupyter_http_over_ws
```

```bash
pip install jupyter_http_over_ws # Run this in case it fails
```

2. Open a Jupyter Notebook Connection

```bash
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
```

3. Copy your token. It should begin with `http://localhost:8888`

![Colab Token](../docs/image.png)

4. Open https://colab.research.google.com and create the Notebook you want

5. On the top-right corner, click Connect > Connect to a local runtime

6. Paste your token

7. Enjoy! Now you're running Rust, in Python, in Jupyter, in Colab! 🤯

![Result](../docs/image-1.png)

## 2. Using Colab traditionally (with imported files)

You will want to do this when:

- You have a sh$t computer :)
- You don't need the repo's codebase to run whatever you want to run (unless you import it)
- You need RAM and compute (GPUs)

### Using Google Drive to upload files

Export the entire repo `/aura` into your Google Drive, and you'll be able to access the files from there. Or you can simply upload a Jupyter Notebook file you already have.

```python
from google.colab import drive
drive.mount('/content/drive')
```

<br>
<br>
ps: Make sure to post your notebooks into the repo!
