"""Render known messages from older project databases in English."""

import re


_TRANSLATIONS = {
    "Lote pausado. Os recortes prontos foram salvos.": "Job paused. Completed tiles have been saved.",
    "O Codex atingiu o limite de uso.": "Codex has reached its usage limit.",
    "Falha ao verificar a configuração do Codex CLI.": "Could not check the Codex CLI configuration.",
    "A configuração do Codex CLI está em um formato incompatível.": "The Codex CLI configuration uses an unsupported format.",
    "Não foi possível iniciar o Codex CLI.": "Could not start the Codex CLI.",
    "Entre no Codex CLI com sua conta do ChatGPT usando codex login.": "Run codex login to sign in to the Codex CLI with your ChatGPT account.",
    "A conexão com o Codex CLI foi encerrada.": "The connection to the Codex CLI closed.",
    "A conexão com o Codex CLI foi interrompida.": "The connection to the Codex CLI was interrupted.",
    "O Codex CLI excedeu o tempo de espera.": "The Codex CLI timed out.",
    "O Codex CLI não abriu o turno de edição.": "The Codex CLI did not open an editing thread.",
    "O Codex CLI não iniciou o turno de edição.": "The Codex CLI did not start the editing turn.",
    "O turno devolveu mais de uma edição para o mesmo recorte.": "The turn returned more than one edit for the same tile.",
    "O turno terminou sem uma edição do ImageGen.": "The turn ended without an ImageGen edit.",
    "O ImageGen devolveu uma edição de outro recorte.": "ImageGen returned an edit for a different tile.",
    "O evento do ImageGen não forneceu um arquivo de imagem acessível.": "The ImageGen event did not provide an accessible image file.",
    "O Codex CLI não concluiu a edição.": "The Codex CLI did not complete the edit.",
    "ImageGen: falha": "ImageGen: failed",
    "Escolha uma imagem estática.": "Choose a still image.",
    "O quadro de edição devolvido pelo ImageGen precisa ser opaco.": "The image returned by ImageGen must be opaque.",
    "O arquivo devolvido pelo ImageGen está inválido.": "ImageGen returned an invalid image file.",
    "O recorte salvo tem um formato inválido.": "The saved tile has an invalid format.",
    "Confira o alinhamento com o original.": "Check alignment with the original.",
    "Confira as cores e os elementos do recorte.": "Check the tile's colors and features.",
    "Há uma diferença visível de cor ou forma na sobreposição.": "Colors or shapes differ visibly in the overlap.",
    "Confira a continuidade dos contornos na sobreposição.": "Check whether contours line up across the overlap.",
    "O arquivo deste recorte está ausente. Gere-o novamente.": "This tile file is missing. Regenerate the tile.",
    "Este recorte já foi substituído por outra edição.": "This tile has already been replaced by another edit.",
}
_SAVED_MESSAGES = re.compile("|".join(map(re.escape, _TRANSLATIONS)))
_TILE_LABEL = re.compile(r"^Coluna (\d+), linha (\d+)\.")
_IMAGE_SIZE = re.compile(
    r"O ImageGen devolveu (\d+×\d+) px\. O recorte precisa ser quadrado, com pelo menos (\d+) px por lado\."
)
_SUBAGENT_COUNT = re.compile(r"Os subagentes entregaram (\d+/\d+) recortes verificáveis\.")


def translate_saved_message(message: str | None) -> str | None:
    if not message:
        return message
    message = _SAVED_MESSAGES.sub(lambda match: _TRANSLATIONS[match[0]], message)
    message = _TILE_LABEL.sub(r"Column \1, row \2.", message)
    message = _IMAGE_SIZE.sub(r"ImageGen returned \1 px. The tile must be square and at least \2 px on each side.", message)
    return _SUBAGENT_COUNT.sub(r"The subagents returned \1 verifiable tiles.", message)
