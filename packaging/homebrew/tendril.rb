# Template: the release workflow fills in __VERSION__ and the checksums and
# publishes it as Formula/tendril.rb in hetsaraiya/homebrew-tap.
class Tendril < Formula
  desc "Plan and run LLMs across the machines you already have"
  homepage "https://github.com/hetsaraiya/homebrew-tap"
  version "__VERSION__"
  license "Apache-2.0"

  base = "https://github.com/hetsaraiya/homebrew-tap/releases/download/tendril-v#{version}"

  on_macos do
    on_arm do
      url "#{base}/tendril-v#{version}-macos-arm64.tar.gz"
      sha256 "__MACOS_ARM64_SHA256__"
    end
    on_intel do
      url "#{base}/tendril-v#{version}-macos-x86_64.tar.gz"
      sha256 "__MACOS_X86_64_SHA256__"
    end
  end

  on_linux do
    on_intel do
      url "#{base}/tendril-v#{version}-linux-x86_64.tar.gz"
      sha256 "__LINUX_X86_64_SHA256__"
    end
  end

  def install
    bin.install "tendril"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/tendril --version")
  end
end
