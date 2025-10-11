HEADLESS = false
# get options from command line
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless)
      HEADLESS=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--headless]"
      exit 0
      ;;
    --) # end of options
      echo sus
      shift
      break
      ;;
    *)
      break
  esac
done

if [ "$HEADLESS" = true ]; then
  echo "Running in headless mode"
  python scripts/random_agent.py --task=Fre25-Isaaclabsym-Direct-v0 --num_envs 128 --headless
else
  echo "Running in GUI mode"
  python scripts/random_agent.py --task=Fre25-Isaaclabsym-Direct-v0 --num_envs 2
fi
