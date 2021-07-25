    private void goodB2G() throws Throwable
    {
        String data;

        while (true)
        {
            /* POTENTIAL FLAW: data is null */
            data = null;
            break;
        }

        while (true)
        {
            /* FIX: validate that data is non-null */
            if (data != null)
            {
                IO.writeLine("" + data.length());
            }
            else
            {
                IO.writeLine("data is null");
            }
            break;
        }
    }
